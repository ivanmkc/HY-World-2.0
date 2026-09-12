import json
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .general_utils import load_video, get_last_video_frame, adjust_image_size, rank0_log

try:
    from ..models.camera import get_camera_embedding, matrix_to_quaternion, unified_camera_normalization
except ImportError:
    from models.camera import get_camera_embedding, matrix_to_quaternion, unified_camera_normalization
from glob import glob

def assign_scale(h, w, scale_map=None):
    # Generation follows the render, when the run asks it to.
    #
    # Everything below picks a generation size by ASPECT RATIO alone: the scale_map holds
    # the sizes the checkpoint was trained at, and argmin over |ratio difference| throws
    # the requested pixel count away entirely. At upstream's --splitted_resolution 480
    # that is invisible, because traj_generate.py:540-542 renders 480x832 and 480x832 is
    # itself a rung -- the map returns the size it was handed, and nothing reveals that it
    # was never looking at the size.
    #
    # Above 480 it stops being invisible, and it fails the same way at every rung:
    #
    #   render 576x1024 (ratio 0.5625) -> 480x832
    #   render 640x1120 (ratio 0.5714) -> 480x832
    #   render 720x1280 (ratio 0.5625) -> 480x832
    #
    # because 480x832's 0.5769 is nearer each of those than any other entry is. The map
    # itself is not in this file -- it comes from the checkpoint config as cfg.scale_map --
    # but it demonstrably CONTAINS 480x832, because the fallback map below does not, and
    # under that one a 480 run would generate 480x768 against a 480x832 pano_bank and fail
    # the same concatenate on width. Two 480 runs have completed. So raising
    # --splitted_resolution buys no texture detail whatsoever, and it corrupts the memory
    # bank on the way past: update_memory (retrieval_wm.py:1210) files those 480-tall
    # generated frames beside the 720-tall pano_bank frames, and the NEXT trajectory's
    # retrieval dies in np.concatenate (retrieval_wm.py:1158) with "the array at index 0
    # has size 480 and the array at index 5 has size 720". That is run
    # worldstereo2-20260912-011304 -- all four ranks, after traj, render and captions had
    # already succeeded at 720x1280.
    #
    # Resizing the frames to agree would be the wrong repair twice over. It would hide a
    # size the model was never asked to produce, and the texture ceiling -- the only
    # reason anyone raises --splitted_resolution -- would not move by one pixel, which is
    # a run that costs four A100-hours to produce the result it already had.
    #
    # So generate at the rendered size instead. The bank is then homogeneous by
    # construction rather than by the coincidence that 480x832 is a rung, and the
    # intrinsic rescale at line 190 below becomes the identity instead of an 832/1280
    # shrink applied to frames whose cameras the memory bank goes on holding at render
    # scale. adjust_image_size is the same rounding traj_generate.py:542 already applied,
    # so for anything this pipeline rendered it returns the size unchanged; it is called
    # rather than assumed because the transformer needs (h/16)*(w/16) divisible by 8 and
    # a hand-set size need not satisfy that.
    #
    # Off by default, because the ladder also bounds the ABSOLUTE size and that still
    # matters on the non-panorama path at line 152, where the input is an arbitrary image
    # rather than something traj_generate sized. Turning it on is out-of-distribution for
    # a checkpoint trained on the ladder, and it is paid for in VRAM: by the same
    # rounding, 480 is 1560 tokens/frame, 576 is 2304, 640 is 2800 and 720 is 3600.
    if os.environ.get("WORLDSTEREO_NATIVE_SCALE", "0") != "0":
        native_h, native_w = adjust_image_size(h, w)
        rank0_log(f"Native scale: generating at {native_h}x{native_w} for a {h}x{w} render.")
        return [native_h, native_w]

    if scale_map is None:
        scale_map = [[480, 768], [512, 720], [576, 640], [608, 608], [640, 576], [720, 512], [768, 480]]
    hw_ratio = h / w
    map_ratios = [h_ / w_ for h_, w_ in scale_map]
    best_scale_idx = int(np.argmin(np.abs(np.array(map_ratios) - hw_ratio)))
    best_scale = scale_map[best_scale_idx]

    return best_scale


def sort_trajs(scene_path):
    # Generation order: (view*: up --> right --> left) --> target* --> reconstruct* --> wonder*
    view_paths = glob(f"{scene_path}/view*")
    view_paths.sort()
    view_list = []
    for view_path in view_paths:
        if os.path.exists(f"{view_path}/valid_analysis.json"):
            v = json.load(open(f"{view_path}/valid_analysis.json"))
            if v["valid"] != "yes":  # skip invalid views
                continue
        traj_paths_ = glob(f"{view_path}/traj*/render.mp4")
        traj_paths_.sort()
        traj_paths = []
        if f"{view_path}/traj2/render.mp4" in traj_paths_:
            traj_paths.append(f"{view_path}/traj2/render.mp4")
        if f"{view_path}/traj0/render.mp4" in traj_paths_:
            traj_paths.append(f"{view_path}/traj0/render.mp4")
        if f"{view_path}/traj1/render.mp4" in traj_paths_:
            traj_paths.append(f"{view_path}/traj1/render.mp4")
        view_list.extend(traj_paths)
    target_list = glob(f"{scene_path}/target*/traj*/render.mp4")
    target_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    recon_list = glob(f"{scene_path}/reconstruct*/traj0/render.mp4")
    recon_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    wander_list = glob(f"{scene_path}/wonder*/traj*/render.mp4")
    wander_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    total_list = view_list + target_list + recon_list + wander_list

    # If iterative trajectories (reconstruct_*/traj1) exist, place them at the end
    iter_list = glob(f"{scene_path}/reconstruct*/traj1/render.mp4")
    iter_list.sort(key=lambda x: (x.split('/')[-3], x.split('/')[-2]))
    total_list = total_list + iter_list

    # Rename any paths whose 3rd-to-last directory contains "-" by replacing it with "_"
    total_list = rename_hyphen_to_underscore(total_list)

    return total_list


def rename_hyphen_to_underscore(path_list):
    """
    Check whether the 3rd-to-last directory component (i.e. the view*/target*/reconstruct*/wonder* level)
    contains a hyphen. If so, rename that directory by replacing '-' with '_' and update the path list.
    """
    new_path_list = []
    for old_path in path_list:
        parts = old_path.split('/')
        dir_name = parts[-3]  # 3rd-to-last directory name
        if '-' not in dir_name:
            new_path_list.append(old_path)
            continue

        new_dir_name = dir_name.replace('-', '_')
        # Build old and new directory paths
        parent_path = '/'.join(parts[:-3])
        old_dir_path = os.path.join(parent_path, dir_name) if parent_path else dir_name
        new_dir_path = os.path.join(parent_path, new_dir_name) if parent_path else new_dir_name
        if os.path.exists(old_dir_path) and not os.path.exists(new_dir_path):
            os.rename(old_dir_path, new_dir_path)

        # Update directory name in path
        parts[-3] = new_dir_name
        new_path_list.append('/'.join(parts))
    return new_path_list


def load_mutli_traj_dataset(cfg, input_path, output_path, view_id, traj_id, device, ref_index=None, model_type=None, task_type="panorama"):
    if view_id.startswith("reconstruct_") and traj_id == "traj1":  # iterative traj
        pil_image = get_last_video_frame(f"{output_path}/{view_id}/traj0/{model_type}_result.mp4")
        pil_image = Image.fromarray(pil_image)
    else:
        if task_type == "panorama":
            pil_image = Image.open(f"{input_path}/{view_id}/start_frame.png")
        else:
            pil_image = Image.open(f"{input_path}/image.png")
    origin_w, origin_h = pil_image.size
    height, width = assign_scale(origin_h, origin_w, scale_map=cfg.scale_map)
    pil_image = pil_image.resize((width, height), Image.Resampling.BICUBIC)
    meta = {"image": pil_image}

    image = transforms.ToTensor()(pil_image) * 2 - 1

    # load conditions
    render_frames = load_video(f"{input_path}/{view_id}/{traj_id}/render.mp4")
    render_video = torch.stack([transforms.ToTensor()(frame) for frame in render_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    render_video = F.interpolate(render_video, size=(height, width), mode='bicubic')[None]
    # replace the first frame with the high-quality reference image
    render_video[0, 0] = image
    render_video = torch.clip(render_video, -1, 1)  # [-1~1], [1,f,c,h,w]
    render_mask_frames = load_video(f"{input_path}/{view_id}/{traj_id}/render_mask.mp4")
    render_mask = torch.stack([transforms.ToTensor()(frame) for frame in render_mask_frames], dim=0)[:, 0:1]  # [f,1,h,w]
    render_mask = F.interpolate(render_mask, size=(height, width), mode='nearest')[None]  # [0,1],[1,f,1,h,w]
    if view_id.startswith("reconstruct_") and traj_id == "traj1":  # iterative traj
        render_mask[0, 0] = 0
    render_video = einops.rearrange(render_video, "b f c h w -> b c f h w")
    render_mask = einops.rearrange(render_mask, "b f c h w -> b c f h w")
    render_mask[render_mask < 0.5] = 0
    render_mask[render_mask >= 0.5] = 1
    meta["render_video"] = render_video.to(device)
    meta["render_mask"] = render_mask.to(device)

    if meta["render_video"].shape[2] > cfg.nframe:
        indices = np.linspace(0, meta["render_video"].shape[2] - 1, cfg.nframe, dtype=int)
        meta["render_video"] = meta["render_video"][:, :, indices]
    if meta["render_mask"].shape[2] > cfg.nframe:
        indices = np.linspace(0, meta["render_mask"].shape[2] - 1, cfg.nframe, dtype=int)
        meta["render_mask"] = meta["render_mask"][:, :, indices]

    # load camera
    cam_info = json.load(open(f"{input_path}/{view_id}/{traj_id}/camera.json"))
    w2cs = torch.tensor(np.array(cam_info["extrinsic"]), dtype=torch.float32, device=device)
    intrinsic = torch.tensor(np.array(cam_info["intrinsic"]), dtype=torch.float32, device=device)
    intrinsic[:, 0, :] = intrinsic[:, 0, :] / origin_w * width
    intrinsic[:, 1, :] = intrinsic[:, 1, :] / origin_h * height

    if w2cs.shape[0] > cfg.nframe:
        indices = np.linspace(0, w2cs.shape[0] - 1, cfg.nframe, dtype=int)
        w2cs = w2cs[indices]
    if intrinsic.shape[0] > cfg.nframe:
        indices = np.linspace(0, intrinsic.shape[0] - 1, cfg.nframe, dtype=int)
        intrinsic = intrinsic[indices]

    camera_embedding = get_camera_embedding(intrinsic, w2cs, intrinsic.shape[0], height, width, normalize=True, is_w2c=True)

    if os.path.exists(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}_ref_w2cs.json"):
        extrinsic_ref = json.load(open(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}_ref_w2cs.json"))
        extrinsic_ref = torch.tensor(np.array(extrinsic_ref), dtype=torch.float32, device=device)

        # Unified normalization: coordinate system reset based on all cameras,
        extrinsic, extrinsic_ref = unified_camera_normalization(w2cs, extrinsic_ref)

        # Convert R matrix (w2c) to quaternion and concat with t vector
        R_matrix = extrinsic[:, :3, :3]
        t_vector = extrinsic[:, :3, 3]
        quaternion = matrix_to_quaternion(R_matrix)
        camera_qt = torch.cat([quaternion, t_vector], dim=-1).unsqueeze(0)  # [1, f, 7]

        R_matrix_ref = extrinsic_ref[:, :3, :3]
        t_vector_ref = extrinsic_ref[:, :3, 3]
        quaternion_ref = matrix_to_quaternion(R_matrix_ref)
        camera_qt_ref = torch.cat([quaternion_ref, t_vector_ref], dim=-1).unsqueeze(0)  # [1, f_ref, 7]

        meta["camera_qt"] = camera_qt.to(device)
        meta["camera_qt_ref"] = camera_qt_ref.to(device)

    meta["camera_embedding"] = camera_embedding.to(device)
    meta["extrinsics"] = w2cs.to(device)
    meta["intrinsics"] = intrinsic.to(device)

    # load reference
    combined_frames = load_video(f"{output_path}/{view_id}/{traj_id}/memory_inputs/{model_type}.mp4")
    combined_frames = [np.array(frame) for frame in combined_frames]
    reference_video = torch.stack([transforms.ToTensor()(frame[:, :]) for frame in combined_frames], dim=0) * 2 - 1.0  # [f,c,h,w]
    reference_video = F.interpolate(reference_video, size=(height, width), mode='bicubic')

    # Filter frames based on ref_index if provided
    if ref_index is not None and len(ref_index) > 0:
        reference_video = reference_video[ref_index]

    meta["reference_video"] = einops.rearrange(reference_video[None], "b f c h w -> b c f h w").to(device)
    meta["ref_index"] = ref_index

    # load text
    if task_type == "panorama":
        prompt = json.load(open(f"{input_path}/{view_id}/{traj_id}/traj_caption.json"))["prompt"]
    else:
        prompt = json.load(open(f"{input_path}/prompt.json"))["prompt"]
    meta["prompt"] = prompt

    meta["width"] = width
    meta["height"] = height

    return meta
