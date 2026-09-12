"""
WorldStereo unified inference class.

Bundles all sub-models (transformer, text/image encoders, VAE) and the
matching inference pipeline under a single diffusers-style interface::

    worldstereo = WorldStereo.from_pretrained(
        "/path/to/checkpoint_root",
        device=device,
    )
    output = worldstereo(**pipeline_inputs)

Hugging Face format expects ``config.json`` plus ``model.safetensors``
in the same directory.

The config must include a ``model_type`` field with one of the
supported values:

* ``worldstereo-camera``      – keyframe + camera control
* ``worldstereo-memory``      – keyframe + camera control + GGM + SSM
* ``worldstereo-memory-dmd``  – DMD (distribution matching distillation) mode
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import types
from typing import Any

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy

from .attention import WanAttnProcessorSP
from .dmd_scheduler import FlowGeneratorScheduler
from .pipelines.pipeline_dmd_keyframe import RefKFDMDGeneratorPipeline
from .pipelines.pipeline_pcd_keyframe import KFPCDControllerPipeline
from .pipelines.pipeline_ref_keyframe import KFPCDControllerRefPipeline
from .worldstereo import WorldStereoModel, WorldStereoRefSModel

try:
    from ..src.general_utils import rank0_log
except ImportError:
    from src.general_utils import rank0_log

# ── suppress noisy third-party logs ───────────────────────────────────
import logging
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"

# transformers / diffusers print a wall of "Some weights were not
# initialized / unexpected keys" on every load.  We already inspect
# load_state_dict results ourselves in worldstereo_wrapper.py, so
# silence their own reporting.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("diffusers.modeling_utils").setLevel(logging.ERROR)

# huggingface_hub HTTP request logs (newer versions use httpx as the HTTP client)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()

# torch.compile / inductor verbose output
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

# misc deprecation / user warnings from HF internals
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
# ──────────────────────────────────────────────────────────────────────

SUPPORTED_MODEL_TYPES = ("worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd")


def _env_on(name: str, default: str = "0") -> bool:
    """Read a WORLDSTEREO_* switch. Absent means the behaviour this file had before it existed."""
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def _fsdp_cpu_offload(component: str) -> OffloadPolicy:
    """Keep this component's FSDP shard in pinned host RAM instead of on the card.

    WORLDSTEREO_FSDP_CPU_OFFLOAD is a comma-separated list of t5, clip, transformer.
    CPUOffloadPolicy copies the sharded parameters host-to-device immediately before each
    all-gather, so the shard costs nothing on the card between forwards -- which is the
    whole point for the text encoder, whose 2.64 GiB shard is touched once per trajectory
    and then sits through fifteen seconds of denoising.

    Empty by default, because it buys memory with PCIe bandwidth and the transformer entry
    in particular pays 8.72 GB of host-to-device traffic on every one of the four DMD steps.
    """
    wanted = {p.strip().lower() for p in os.environ.get("WORLDSTEREO_FSDP_CPU_OFFLOAD", "").split(",")}
    if component in wanted:
        rank0_log(f"FSDP CPU offload enabled for {component}; its shard will live in pinned host RAM.")
        return CPUOffloadPolicy(pin_memory=True)
    return OffloadPolicy()


def report_memory(label: str) -> None:
    """Print the card's real high-water mark, because every number below was inferred.

    The budget for this stage was assembled from safetensors headers and two OOM messages,
    not from the device: nvidia-smi's once-a-minute heartbeat says 40,433 MiB of 40,960 and
    nothing about what is in it. WORLDSTEREO_MEM_REPORT=1 turns torch's own accounting on at
    the four points that matter, which is what will settle the decomposition on the next run.

    max_memory_allocated is reset at each label, so each line is the peak of the phase that
    just ended rather than the peak of the run so far.
    """
    if not _env_on("WORLDSTEREO_MEM_REPORT") or not torch.cuda.is_available():
        return
    gib = 1024**3
    rank0_log(
        f"[mem] {label}: allocated={torch.cuda.memory_allocated() / gib:.2f} GiB "
        f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB "
        f"peak_since_last={torch.cuda.max_memory_allocated() / gib:.2f} GiB"
    )
    torch.cuda.reset_peak_memory_stats()


@contextlib.contextmanager
def parked_on_cpu(modules, *, device):
    """Park models on the host for the duration of a block that does not call them.

    Two perception models are built once and then left on the card for the whole stage,
    although neither is reachable from the diffusion pipeline:

      MoGe-2 ViT-L      video_gen.py:104, used only by memory_bank.alignment()
                        (retrieval_wm.py:1606), which runs AFTER every trajectory is
                        generated. model.pt is 1,323,815,904 bytes of fp32 -> 1.23 GiB.
      dinov2-base       CameraSelector._load_model, retrieval_wm.py:489, used only by
                        memory_bank.retrieval() (retrieval_wm.py:1106), which runs
                        immediately BEFORE the pipeline call and not during it.
                        86.6M fp32 -> 0.32 GiB.

    Neither is sharded, so every rank pays the full price, and 1.55 GiB of a 39.49 GiB card
    is idle through the entire denoise loop. Moving them out and back costs 1.67 GB each way
    over PCIe per trajectory -- about 0.17 s against the ~25 s a trajectory takes.

    Enabled by WORLDSTEREO_IDLE_MODEL_OFFLOAD=1. Off by default; it cannot change any output,
    but nor should a memory saving arrive without a flag to attribute it to.
    """
    live = [m for m in modules if m is not None] if _env_on("WORLDSTEREO_IDLE_MODEL_OFFLOAD") else []
    for module in live:
        module.to("cpu")
    if live:
        torch.cuda.empty_cache()
    try:
        yield
    finally:
        for module in live:
            module.to(device)


def _get_half_dtype() -> torch.dtype:
    """Select the best half-precision dtype based on current GPU capability: bf16 > fp16 > fp32."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        return torch.float16
    else:
        return torch.float32


class WorldStereo:
    """Diffusers-style wrapper that owns every sub-model and its pipeline."""

    def __init__(self, pipeline: Any, cfg: Any) -> None:
        self.pipeline = pipeline
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        subfolder: str = "",
        local_files_only: bool = False,
        sp_world_size: int = 1,
        fsdp: bool = False,
        device_mesh=None,
        device: torch.device | None = None,
    ) -> "WorldStereo":
        """
        Build a WorldStereo instance from Hugging Face format
        (``config.json`` + ``model.safetensors``).

        Args:
            repo_id: Model directory or HF repo ID.
            subfolder: Subfolder within the HF repo or local directory. This is equivalent to the `model_type` (e.g., 'worldstereo-camera').
            local_files_only: If True, avoid downloading the file and return the path to the local cached file if it exists.
            sp_world_size: Sequence-Parallel degree (1 = disabled).
            fsdp: Wrap models with PyTorch FSDP.  Requires ``device_mesh``.
            device_mesh: ``DeviceMesh`` with dims ``("rep", "shard")``.
            device: Target CUDA device.
        """
        if os.path.isdir(repo_id):
            json_cfg_path = os.path.join(repo_id, subfolder, "config.json")
            safetensors_path = os.path.join(repo_id, subfolder, "model.safetensors")

            if not os.path.exists(json_cfg_path):
                raise FileNotFoundError(f"config.json not found under {json_cfg_path!r}")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"model.safetensors not found at {safetensors_path!r}")
        else:
            from huggingface_hub import hf_hub_download

            json_cfg_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )
            safetensors_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                subfolder=subfolder if subfolder else None,
                local_files_only=local_files_only,
            )

        cfg = OmegaConf.create(cls._load_hf_config(json_cfg_path))
        model_weights_path = safetensors_path

        model_type = subfolder
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model_type {model_type!r}. Expected one of {SUPPORTED_MODEL_TYPES}.")

        transformer = cls._load_transformer(
            cfg,
            model_type,
            model_weights_path,
            sp_world_size=sp_world_size,
            fsdp=fsdp,
            device_mesh=device_mesh,
            device=device,
            local_files_only=local_files_only,
        )

        text_encoder, image_clip, vae = cls._load_aux(
            cfg, device=device, device_mesh=device_mesh, fsdp=fsdp, local_files_only=local_files_only
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            cfg.base_model, do_rescale=False, subfolder="image_processor", local_files_only=local_files_only
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, subfolder="tokenizer", local_files_only=local_files_only
        )

        pipeline = cls._build_pipeline(
            model_type,
            cfg,
            transformer=transformer,
            text_encoder=text_encoder,
            image_clip=image_clip,
            image_processor=image_processor,
            tokenizer=tokenizer,
            vae=vae,
            device=device,
            local_files_only=local_files_only,
        )

        rank0_log(f"WorldStereo ({model_type}) ready.")
        return cls(pipeline=pipeline, cfg=cfg)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward all arguments to the underlying inference pipeline."""
        return self.pipeline(*args, **kwargs)

    def to(self, device: torch.device) -> "WorldStereo":
        self.pipeline = self.pipeline.to(device)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_config(config_json_path: str) -> dict[str, Any]:
        with open(config_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        required_keys = ["base_model", "controlnet_cfg"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(
                f"config.json missing required keys: {missing}. "
                "Please use the conversion script to export a valid HF package."
            )

        return cfg

    @staticmethod
    def _load_transformer(
        cfg,
        model_type: str,
        weights_path: str,
        *,
        sp_world_size: int,
        fsdp: bool,
        device_mesh,
        device,
        local_files_only: bool = False,
    ):

        half_dtype = _get_half_dtype()
        rank0_log(f"Loading transformer ({model_type})… dtype={half_dtype}")

        # local_files_only is threaded through every other from_pretrained in this file and
        # was missing from these two -- which are the largest. Wan2.1's transformer subfolder
        # is 65.58 GB, so without it four torchrun ranks each resolve and fetch that through
        # the Hub API at once: roughly 262 GB of concurrent download behind huggingface_hub's
        # per-repo lock, printing nothing while it happens. That is the stall that consumed
        # several multi-hour runs before HF_HUB_OFFLINE turned it into a named error.
        if model_type == "worldstereo-camera":
            transformer = WorldStereoModel.from_pretrained(
                cfg.base_model,
                subfolder="transformer",
                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
                local_files_only=local_files_only,
            )
        else:
            transformer = WorldStereoRefSModel.from_pretrained(
                cfg.base_model,
                subfolder="transformer",
                controlnet_cfg=cfg.controlnet_cfg,
                torch_dtype=half_dtype,
                local_files_only=local_files_only,
            )

        rank0_log("Building ControlNet…")
        transformer.build_controlnet(load_uni3c=False, freeze_backbone=cfg.freeze_backbone)

        if sp_world_size > 1:
            transformer.sp_size = sp_world_size
            for layer in transformer.controlnet.controlnet_blocks:
                layer.self_attn.processor.sp_size = sp_world_size
            for block in transformer.blocks:
                if model_type == "worldstereo-camera":
                    block.attn1.set_processor(WanAttnProcessorSP(sp_size=sp_world_size))
                else:
                    block.attn1.processor.sp_size = sp_world_size

        rank0_log(f"Loading HF safetensors weights from {weights_path}…")
        weights = load_safetensors(weights_path, device="cpu")

        result = transformer.load_state_dict(weights, strict=False)

        # Release the checkpoint before FSDP wraps anything.
        #
        # load_state_dict copies into the model's existing parameters, so once it returns
        # this dict is a second, redundant full copy of the checkpoint in host RAM --
        # 34.86 GB of it for worldstereo-memory-dmd. Holding it until the function returns
        # means every rank carries backbone + checkpoint simultaneously through the FSDP
        # wrap, which is ~68 GB per rank; four ranks peaked at 315.4 GB of a 334 GB host
        # and were OOM-killed. Dropping it here is what makes four ranks affordable, and
        # four ranks is what makes the shard small enough to leave room on a 40 GB card.
        #
        # Nothing below reads it: _summarize_keys works from transformer.state_dict(), and
        # the FSDP branch only touches the module.
        del weights
        gc.collect()

        def _summarize_keys(keys: list[str], label: str) -> None:
            if not keys:
                return
            from collections import Counter

            # Count unloaded parameters
            total_params = sum(transformer.state_dict()[k].numel() for k in keys if k in transformer.state_dict())
            # Count occurrence frequency of each field (split by ".") across all keys, take top-2
            field_counter: Counter[str] = Counter()
            for k in keys:
                parts = k.split(".")
                # Skip pure numeric indices (e.g. blocks.0) and common prefixes/suffixes
                field_counter.update(p for p in parts if not p.isdigit())
            top_fields = [f for f, _ in field_counter.most_common(2)]
            # Filter representative keys using top-2 fields (prefer keys that contain both fields)
            repr_keys = sorted([k for k in keys if all(f in k.split(".") for f in top_fields)])
            if not repr_keys:
                repr_keys = sorted(keys)
            sample_keys = repr_keys[:3]
            rank0_log(
                f"{label}: {len(keys)} keys ({total_params / 1e6:.1f}M params), "
                f"top fields: {top_fields}. "
                f"Representative: {sample_keys}"
                + (f" … and {len(keys) - len(sample_keys)} more" if len(keys) > len(sample_keys) else "")
            )
            rank0_log(f"These are frozen backbone weights initialized by the base video model ({cfg.base_model}).")

        _summarize_keys(result.unexpected_keys, "Unexpected keys")
        _summarize_keys(result.missing_keys, "Missing keys")

        # Out of autograd before the wrap, not after.
        #
        # This stage never calls backward: the DMD pipeline runs mode="test", which takes
        # the torch.no_grad() branch at pipeline_dmd_keyframe.py:266 for every step. But
        # requires_grad is still True on the controlnet (build_controlnet re-enables it at
        # worldstereo.py:153), and requires_grad is exactly what decides whether autocast
        # CACHES its down-cast of a parameter -- see the note in _load_aux, where the same
        # switch is worth 9.80 GiB. Nothing here is fp32 under a bf16 autocast, so the
        # transformer itself has nothing to cache; it is set for uniformity and because a
        # mixed-requires_grad FSDP group is a foot-gun nobody needs.
        #
        # Before fully_shard, because FSDP copies requires_grad onto the unsharded parameter
        # when it first materialises it (_fsdp_param.py:537).
        if _env_on("WORLDSTEREO_INFER_NO_GRAD"):
            transformer.requires_grad_(False)
            rank0_log("Transformer parameters set requires_grad=False (inference only).")

        if fsdp:
            fsdp_kwargs = dict(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=half_dtype,
                    reduce_dtype=torch.float32,
                ),
                mesh=device_mesh["rep", "shard"],
                reshard_after_forward=True,
                offload_policy=_fsdp_cpu_offload("transformer"),
            )
            transformer = transformer.to(half_dtype)
            for layer in transformer.blocks:
                fully_shard(layer, **fsdp_kwargs)
            for layer in transformer.controlnet.controlnet_blocks:
                fully_shard(layer, **fsdp_kwargs)
            fully_shard(transformer, **fsdp_kwargs)
            rank0_log("FSDP wrapping done for transformer.")
        else:
            transformer = transformer.to(device=device)

        gc.collect()
        torch.cuda.empty_cache()
        report_memory("transformer loaded")
        return transformer.eval()

    @staticmethod
    def _load_aux(cfg, *, device, device_mesh, fsdp: bool, local_files_only: bool = False):
        import transformers as _tr
        from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

        # ---- text encoder ----
        # float32 by default, as upstream has it. WORLDSTEREO_TEXT_ENCODER_DTYPE=bfloat16
        # halves it, which matters on a host with more ranks than headroom: the weights are
        # 22.72 GB on disk and upcasting to fp32 makes them roughly 45 GB resident, so four
        # ranks want ~180 GB of a 340 GB machine before the transformer, the VAE and MoGe
        # are counted. A run wedged for two hours at exactly this line is what prompted it.
        # Opt-in, because a T5 encoder is the one part of a pipeline like this with a real
        # reason to stay in fp32.
        _text_dtype = getattr(torch, os.environ.get("WORLDSTEREO_TEXT_ENCODER_DTYPE", "float32"))
        rank0_log(f"Loading TextEncoder (UMT5) in {_text_dtype}…")
        text_encoder = UMT5EncoderModel.from_pretrained(
            cfg.base_model, subfolder="text_encoder", torch_dtype=_text_dtype, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching text_encoder.encoder.embed_tokens for transformers>=5.0.0", "WARNING")
            text_encoder.encoder.embed_tokens = text_encoder.shared
        text_encoder = torch.compile(text_encoder)

        # ---- image encoder ----
        # float32 by default, as upstream has it, and as upstream has it for no reason: Wan
        # ships image_encoder/model.safetensors at 1.264 GB for 630.6M parameters, which is
        # fp16 on disk. Loading it fp32 re-inflates it to 2.52 GB and recovers no precision
        # that the checkpoint ever had. WORLDSTEREO_IMAGE_ENCODER_DTYPE=bfloat16 halves both
        # the 0.59 GiB resident shard and the 2.35 GiB all-gather that one CLIP forward does.
        # Opt-in, in the same shape as the text encoder above, because the two are the same
        # decision and should read the same way.
        _clip_dtype = getattr(torch, os.environ.get("WORLDSTEREO_IMAGE_ENCODER_DTYPE", "float32"))
        rank0_log(f"Loading ImageEncoder (CLIP) in {_clip_dtype}…")
        image_clip = CLIPVisionModel.from_pretrained(
            cfg.base_model, subfolder="image_encoder", torch_dtype=_clip_dtype, local_files_only=local_files_only
        ).eval()
        if _tr.__version__ >= "5.0.0":
            rank0_log("Patching CLIP vision forward for transformers>=5.0.0", "WARNING")

            def _clip_vision_forward(self, pixel_values=None, interpolate_pos_encoding=False, **kwargs):
                if pixel_values is None:
                    raise ValueError("pixel_values is required")
                hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
                hidden_states = self.pre_layrnorm(hidden_states)
                encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
                pooled_output = self.post_layernorm(encoder_outputs.last_hidden_state[:, 0, :])
                return BaseModelOutputWithPooling(
                    last_hidden_state=encoder_outputs.last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                )

            def _clip_encoder_forward(self, inputs_embeds, attention_mask=None, **kwargs):
                hidden_states = inputs_embeds
                encoder_states = ()
                for layer in self.layers:
                    encoder_states = encoder_states + (hidden_states,)
                    hidden_states = layer(hidden_states, attention_mask, **kwargs)
                encoder_states = encoder_states + (hidden_states,)
                return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

            image_clip.vision_model.forward = types.MethodType(_clip_vision_forward, image_clip.vision_model)
            image_clip.vision_model.encoder.forward = types.MethodType(
                _clip_encoder_forward, image_clip.vision_model.encoder
            )

        # ---- VAE ----
        vae_dtype = _get_half_dtype()
        rank0_log(f"Loading 3D-VAE… dtype={vae_dtype}")
        vae = AutoencoderKLWan.from_pretrained(
            cfg.base_model, subfolder="vae", torch_dtype=vae_dtype, local_files_only=local_files_only
        ).eval()
        vae = torch.compile(vae)

        # Take the two encoders out of autograd, which is the largest single saving in this
        # file and costs nothing.
        #
        # video_gen.py:284 opens ONE torch.autocast("cuda", bfloat16) around the whole
        # pipeline call. Every autocast region below it is nested, and autocast clears its
        # weight cache only when the OUTERMOST region exits -- so anything cached while
        # encoding the prompt is still on the card through all four denoise steps and the
        # VAE decode.
        #
        # What gets cached is a bf16 copy of every fp32 parameter an autocast-listed op
        # touches, and the condition for caching (ATen's cached_cast) is
        # `arg.requires_grad() && arg.is_leaf() && !arg.is_view()`. FSDP2's unsharded
        # parameter is an nn.Parameter built by _fsdp_param.py:537 -- leaf, not a view, and
        # requires_grad inherited from the sharded one, which is True because
        # from_pretrained().eval() never clears it. So both encoders qualify:
        #
        #   UMT5 block Linears   4.6306B params -> 9.26 GB of bf16 copies -> 8.63 GiB
        #   CLIP  block Linears  0.6291B params -> 1.26 GB                -> 1.17 GiB
        #                                                          total    9.80 GiB
        #
        # per rank, held from encode_prompt to the end of the pipeline call, on a card with
        # 39.49 GiB. That is not a guess about the mechanism: an FSDP2 + autocast rig on CPU
        # reproduces it exactly -- 8 blocks of 67.1M fp32 parameters retain 170.8 MiB inside
        # the region against a 128.0 MiB predicted cache, and retain 0.0 MiB with
        # requires_grad False. It also matches where two runs actually died: ws2e and ws2h
        # both OOMed INSIDE encode_prompt, ws2h on a 738 MiB request, which is exactly one
        # UMT5 block's fp32 all-gather (192.94M x 4 B = 735.9 MiB).
        #
        # requires_grad has no effect on any forward value, so this changes nothing about
        # the output. It is still a knob, and still off by default, because the saving is
        # large enough that it should be attributable to a flag rather than to a release.
        if _env_on("WORLDSTEREO_INFER_NO_GRAD"):
            text_encoder.requires_grad_(False)
            image_clip.requires_grad_(False)
            vae.requires_grad_(False)
            rank0_log("Aux encoders set requires_grad=False; autocast will not cache their weights.")

        if fsdp:
            # param_dtype for the aux encoders.
            #
            # fp32 as upstream has it, which under WORLDSTEREO_TEXT_ENCODER_DTYPE=bfloat16 is
            # a round trip that gains nothing: the sharded weights are already bf16, FSDP
            # upcasts them to fp32 for the all-gather, and the enclosing autocast casts them
            # straight back down for every matmul. The only thing the upcast buys is a
            # doubled all-gather buffer -- 735.9 MiB per UMT5 block rather than 368.0 --
            # and that buffer is the allocation ws2h died on.
            #
            # "match" uses each module's own dtype instead. It is not free of consequence:
            # ops autocast does NOT cast (the RMS norms, the fp32 softmax) would then run in
            # bf16 rather than fp32, so this is a numerical change and not merely a memory
            # one. Wan's own release ships umt5-xxl as an "enc-bf16" checkpoint, so bf16 is
            # defensible -- but it is a second lever, not part of the first one.
            _aux_param_dtype_env = os.environ.get("WORLDSTEREO_AUX_FSDP_PARAM_DTYPE", "float32")

            def _aux_fsdp_kwargs(module, component: str):
                param_dtype = module.dtype if _aux_param_dtype_env == "match" else getattr(torch, _aux_param_dtype_env)
                return dict(
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=param_dtype,
                        reduce_dtype=torch.float32,
                    ),
                    mesh=device_mesh["rep", "shard"],
                    reshard_after_forward=True,
                    offload_policy=_fsdp_cpu_offload(component),
                )

            t5_kwargs = _aux_fsdp_kwargs(text_encoder, "t5")
            for layer in text_encoder.encoder.block:
                fully_shard(layer, **t5_kwargs)
            fully_shard(text_encoder, **t5_kwargs)
            rank0_log(f"FSDP wrapping done for T5 (param_dtype={t5_kwargs['mp_policy'].param_dtype}).")

            clip_kwargs = _aux_fsdp_kwargs(image_clip, "clip")
            for layer in image_clip.vision_model.encoder.layers:
                fully_shard(layer, **clip_kwargs)
            fully_shard(image_clip, **clip_kwargs)
            rank0_log(f"FSDP wrapping done for CLIP (param_dtype={clip_kwargs['mp_policy'].param_dtype}).")

            gc.collect()
            torch.cuda.empty_cache()
        else:
            text_encoder = text_encoder.to(device=device)
            image_clip = image_clip.to(device=device)

        vae = vae.to(device=device)

        # Tile the VAE, because the card is already exactly full at the default resolution.
        #
        # Both successful runs peaked at 40,373 MiB of a 39,490 MiB card during stage
        # video -- not close to the limit, at it. Nothing can be asked of a higher
        # --splitted_resolution until something gives memory back, and the VAE is the
        # cheapest place to find it: diffusers ships AutoencoderKLWan with use_tiling
        # False, so it decodes the whole clip in one piece, which for video is the single
        # largest allocation in the stage.
        #
        # Tiles are 256x256 with a 192 stride, so neighbours overlap by 64 pixels and are
        # blended (blend_v/blend_h) rather than butted together. That matters more here
        # than in a normal pipeline: these frames are not the deliverable, they are what
        # the 3DGS trains on, so a visible seam would be baked into the world rather than
        # merely looked at.
        #
        # On by default. Off via WORLDSTEREO_VAE_TILING=0, which is how to tell a tiling
        # artefact apart from a reconstruction one if the world ever looks wrong in a grid.
        if os.environ.get("WORLDSTEREO_VAE_TILING", "1") != "0":
            vae.enable_tiling()
            rank0_log("VAE tiling enabled (256px tiles, 192 stride).")

        report_memory("aux models loaded")
        return text_encoder, image_clip, vae

    @staticmethod
    def _build_pipeline(
        model_type: str,
        cfg,
        *,
        transformer,
        text_encoder,
        image_clip,
        image_processor,
        tokenizer,
        vae,
        device,
        local_files_only: bool = False,
    ):
        common = dict(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_clip,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
        )
        if model_type == "worldstereo-camera":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory":
            scheduler = UniPCMultistepScheduler.from_pretrained(
                cfg.base_model, subfolder="scheduler", local_files_only=local_files_only
            )
            return KFPCDControllerRefPipeline(**common, scheduler=scheduler)

        if model_type == "worldstereo-memory-dmd":
            scheduler = FlowGeneratorScheduler(
                start_timesteps=cfg.dmd_start_steps,
                num_train_timesteps=cfg.dmd_end_steps,
                shift=cfg.gen_shift,
                use_timestep_transform=True,
                dmd_steps=cfg.dmd_steps,
                rank=dist.get_rank(),
            )
            return RefKFDMDGeneratorPipeline(
                **common,
                scheduler=scheduler,
                device=device,
                vae_compile=False,
                vae_compile_mode="max-autotune",
            )

        raise ValueError(f"Unknown model_type: {model_type!r}")
