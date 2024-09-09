# import os
# from os import makedirs
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torchvision
# from scene import Scene, GaussianModel
# from argparse import ArgumentParser, Namespace
# from arguments import ModelParams, PipelineParams, OptimizationParams
# from utils.loss_utils import l1_loss, ssim, msssim
# from omegaconf import OmegaConf
# from omegaconf.dictconfig import DictConfig
# import numpy as np
# import random
# from tqdm import tqdm
# from time import time
# from gaussian_renderer import render
# import concurrent.futures
# import imageio
# from ip2p_sequence import SequenceInstructPix2Pix
# from ip2p_models.RAFT.raft import RAFT
# from ip2p_models.RAFT.utils.utils import InputPadder
# from einops import rearrange
# import datetime
# import threading
# import math
# import sys
# import cv2
# import ipdb

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# def multithread_write(image_list, path):
#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
#     def write_image(image, count, path):
#         try:
#             torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
#             return count, True
#         except:
#             return count, False
        
#     tasks = []
#     for index, image in enumerate(image_list):
#         tasks.append(executor.submit(write_image, image, index, path))
#     executor.shutdown()
#     for index, status in enumerate(tasks):
#         if status == False:
#             write_image(image_list[index], index, path)

# enhomo0 = lambda x: torch.cat([x, torch.ones([1, *x.shape[1:]], device=x.device, dtype=x.dtype)], dim=0)
# dehomo0 = lambda x: x[:-1] / x[-1:]

# def warp_pts_BfromA(intriA, w2cA, depthA, intriB, w2cB):
#     """
#     Warp pixels from view A to view B.

#     Args:
#         intriA: Intrinsic matrix of view A (3, 3).
#         w2cA: World to camera transformation matrix of view A (4, 4).
#         depthA: Depth map of view A (H, W).
#         intriB: Intrinsic matrix of view B (3, 3).
#         w2cB: World to camera transformation matrix of view B (4, 4).

#     Returns:
#         Corresponding pixel coordinates in view B.
#     """
#     # Get the height and width from the depth map
#     H, W = depthA.shape

#     # Create a meshgrid for the pixel coordinates
#     y, x = torch.meshgrid(
#         torch.arange(H, device=depthA.device, dtype=torch.float32), 
#         torch.arange(W, device=depthA.device, dtype=torch.float32), 
#         indexing='ij'
#     )
#     x = x.flatten()
#     y = y.flatten()

#     # Create pixel coordinates in homogeneous form (3, N)
#     pix_coords_A = torch.stack([x, y, torch.ones_like(x, device=depthA.device)], dim=0)

#     # Convert pixel coordinates to camera coordinates (3, N)
#     cam_coords_A = torch.matmul(torch.inverse(intriA), pix_coords_A) * depthA.flatten()

#     # Convert camera coordinates to world coordinates (4, N)
#     cam_coords_A_homo = enhomo0(cam_coords_A)
#     world_coords_A = torch.matmul(torch.inverse(w2cA), cam_coords_A_homo)

#     # Convert world coordinates to camera coordinates in view B (4, N)
#     cam_coords_B_homo = torch.matmul(w2cB, world_coords_A)

#     # Convert camera coordinates to pixel coordinates in view B (3, N)
#     pix_coords_B_homo = torch.matmul(intriB, dehomo0(cam_coords_B_homo))

#     # Normalize homogeneous coordinates to get final pixel coordinates in view B
#     pix_coords_B = dehomo0(pix_coords_B_homo).reshape(2, H, W)

#     return pix_coords_B.permute(1, 2, 0)


# def apply_warp(
#     AfromB: torch.Tensor, 
#     imgA: torch.Tensor, 
#     imgB: torch.Tensor,
#     u=None, d=None, l=None, r=None, 
#     default=torch.tensor([0, 0, 0]),
#     threshold=0.01
# ):
#     """
#     Warp imgB to imgA based on precomputed correspondence AfromB.
    
#     Args:
#         AfromB: Precomputed correspondence from B to A, shape (H, W, 2).
#         imgA: Target image A, shape (H, W, C).
#         imgB: Source image B, shape (H, W, C).
#         u, d, l, r: Bounds for cropping the image.
#         default: Default color for out-of-boundary pixels or mismatches.
#         threshold: Threshold to filter out less accurate warps.
    
#     Returns:
#         imgA_warped: Warped image with pixels from imgB.
#         mask: Boolean mask indicating valid pixels.
#     """
#     # Set boundaries
#     u, d, l, r = u or 0, d or imgA.shape[0], l or 0, r or imgA.shape[1]
#     default = default.to(dtype=imgA.dtype, device=imgA.device)
    
#     # Extract X and Y coordinates from AfromB
#     Y, X = AfromB.permute(2, 0, 1)
    
#     # Calculate the mask for valid coordinates within bounds
#     mask = (u <= X) & (X < d) & (l <= Y) & (Y < r)
    
#     # 引入一个基于阈值的附加过滤条件，过滤掉不准确的映射点
#     delta = torch.sqrt((X - torch.round(X)) ** 2 + (Y - torch.round(Y)) ** 2)
#     mask &= delta < threshold
    
#     # Normalize the coordinates to [-1, 1] for grid_sample
#     X_norm = ((X - u) / (d - 1 - u) * 2 - 1) * mask
#     Y_norm = ((Y - l) / (r - 1 - l) * 2 - 1) * mask
#     pix = torch.stack([Y_norm, X_norm], dim=-1).unsqueeze(0)  # shape (1, H, W, 2)
    
#     # Warp imgB to imgA using grid_sample
#     imgA_warped = F.grid_sample(imgB.permute(2, 0, 1).unsqueeze(0), pix, mode='bilinear', align_corners=True)
#     imgA_warped = imgA_warped.squeeze(0).permute(1, 2, 0)
    
#     # Apply the mask to imgA_warped, setting invalid pixels to default color
#     imgA_warped[~mask] = default
    
#     return imgA_warped, mask


# def warp_flow(img, flow): 
#     # warp image according to flow
#     h, w = flow.shape[:2]
#     flow_new = flow.copy() 
#     flow_new[:, :, 0] += np.arange(w) 
#     flow_new[:, :, 1] += np.arange(h)[:, np.newaxis] 

#     res = cv2.remap(
#         img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
#     )
#     return res


# def compute_bwd_mask(fwd_flow, bwd_flow):
#     # compute the backward mask
#     alpha_1 = 0.5 
#     alpha_2 = 0.5

#     fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
#     bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

#     bwd_mask = (
#         bwd_lr_error
#         < alpha_1
#         * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
#         + alpha_2
#     )

#     return bwd_mask

    

# def gaussian_editing(args, dataset, opt, pipe):
#     # log
#     tag_prompt = args.prompt.split(' ')[-1].replace('?','')
#     tag = f'{datetime.datetime.now().strftime("%d-%H-%M")}_{tag_prompt}'


#     # load gaussians and scene
#     gaussians = GaussianModel(args.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if args.eval_shfs_4d else 0)
#     scene = Scene(dataset, gaussians, shuffle=False, num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
#     (model_param, _) = torch.load(args.chkpnt_path)
#     gaussians.restore(model_param, opt)

#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
#     if pipe.env_map_res:
#         env_map = nn.Parameter(torch.zeros((3, pipe.env_map_res, pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(False))
#         env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
#     else:
#         env_map = None

#     gaussians.env_map = env_map

#     num_cam = len([f for f in os.listdir(args.source_path) if f.endswith('.mp4')])

#     # load ip2p, raft
#     ip2p = SequenceInstructPix2Pix(device=args.ip2p_device, ip2p_use_full_precision=args.ip2p_use_full_precision)
    
#     raft = torch.nn.DataParallel(RAFT(args=Namespace(small=False, mixed_precision=False)))
#     raft.load_state_dict(torch.load(args.raft_ckpt))
#     raft = raft.module
    
#     raft = raft.to(args.ip2p_device)
#     raft.requires_grad_(False)
#     raft.eval()
#     print('RAFT loaded!')

#     # Sampler, 根据训练的iteration来选择不同的Sampler
#     # trainingSampler = MotionSampler(allrgbs, args.num_frames, args.batch_size)
#     # simpleSampler = SimpleSampler(args.num_frames, args.batch_size)

    
#     time_cam_dataset = scene.getTrainCameras()  # all_rgbs


#     # cache for depth
#     cache_depth_path = os.path.join(args.cache_path, args.exp_name, 'depth_all.pt')
#     if os.path.exists(cache_depth_path):
#         key_frame_depth_maps = torch.load(cache_depth_path).cpu()
#         print('all depth maps loaded from cache dir')
#         torch.cuda.empty_cache()
#     else:
#         cache_path = os.path.join(args.cache_path, args.exp_name)
#         os.makedirs(cache_path, exist_ok=True)
#         key_frame_depth_maps = []
#         key_frame_dataset = scene.getKeyCameras()

#         with torch.no_grad():
#             for idx, view in enumerate(tqdm(key_frame_dataset, desc="Rendering depth maps for key frames")):
#                 _, viewpoint_cam = view
#                 viewpoint_cam = viewpoint_cam.cuda()

#                 depth_map = render(viewpoint_cam, gaussians, pipe, background)["depth"] # (1, H, W)
#                 key_frame_depth_maps.append(depth_map.squeeze(0))
#             key_frame_depth_maps = torch.stack(key_frame_depth_maps, dim=0) # [cam_num, H, W]
#             torch.save(key_frame_depth_maps, cache_depth_path)
#             print("all key frame's depth maps saved to cache dir")
#             torch.cuda.empty_cache()
    
#     all_rgbs = time_cam_dataset.time_cam_image.to(device) # [num_frame * num_cam, 3, H, W], cuda 0
#     H, W = all_rgbs.shape[-2:]
#     all_rgbs = all_rgbs.view(num_cam, dataset.num_frames, 3, H, W).permute(1, 0, 2, 3, 4) # [num_frame, num_cam, 3, H, W]
#     original_rgbs = all_rgbs.clone() # [num_frame, num_cam, 3, H, W]
#     cam_idxs = list(range(0, num_cam))

#     # thread lock for data and model
#     data_lock = threading.Lock()

#     os.makedirs(f'output/vis/{tag}', exist_ok=True)

#     def key_frame_edit(key_frame:int = 0, warp_ratio:float = 0.5, warm_up_steps:int = 12):
#         print(f'key frame {key_frame} editing')
#         for warm_up_idx in range(warm_up_steps):
#             sample_idxs = sorted(list(np.random.choice(cam_idxs, args.sequence_length, replace=False)))
#             remain_idxs = sorted(list(set(cam_idxs) - set(sample_idxs)))

#             sample_images = all_rgbs[key_frame][sample_idxs] # [sample_length, 3, H, W], cuda 0
#             sample_images_cond = original_rgbs[key_frame][sample_idxs] # [sample_length, 3, H, W], cuda 0

#             remain_images = all_rgbs[key_frame][remain_idxs] # [remain_length, 3, H, W], cuda 0
#             remain_images_cond = original_rgbs[key_frame][remain_idxs] # [remain_length, 3, H, W], cuda 0

#             torchvision.utils.save_image(sample_images, f'output/vis/{tag}/sample_images.png', nrow=sample_images.shape[0], normalize=True)
#             torchvision.utils.save_image(sample_images_cond, f'output/vis/{tag}/sample_images_cond.png', nrow=sample_images_cond.shape[0], normalize=True)

#             # cosine annealing
#             scale_min, scale_max = 0.8, 1.0
#             scale = scale_min + 0.5 * (1 + math.cos(math.pi * warm_up_idx / warm_up_steps)) * (scale_max - scale_min)
#             sample_images_edit = ip2p.edit_sequence(
#                 images=sample_images.unsqueeze(0), # (1, sample_length, C, H, W)
#                 images_cond=sample_images_cond.unsqueeze(0), # (1, sample_length, C, H, W)
#                 guidance_scale=args.guidance_scale,
#                 image_guidance_scale=args.image_guidance_scale,
#                 diffusion_steps=int(args.diffusion_steps * scale),
#                 prompt=args.prompt,
#                 noisy_latent_type="noisy_latent",
#                 T=int(1000 * scale),
#             ) # (1, C, f, H, W), cuda 0

#             sample_images_edit = rearrange(sample_images_edit, '1 C f H W -> f C H W').to(device, dtype=torch.float32) # (f, C, H, W), cuda 0
#             if sample_images_edit.shape[-2:] != (H, W):
#                 sample_images_edit = F.interpolate(sample_images_edit, size=(H, W), mode='bilinear', align_corners=False)

#             torchvision.utils.save_image(sample_images_edit, f'output/vis/{tag}/{warm_up_idx}_sample_images_edit_before.png', nrow=sample_images_edit.shape[0], normalize=True)

#             # spatial warp (based on depth map, see ViCA-NeRF)
#             for idx_cur, i in enumerate(sample_idxs):
#                 warp_average = torch.zeros((H, W, 3), dtype=torch.float32, device=device) # cuda 0
#                 weights_mask = torch.zeros((H, W), dtype=torch.float32, device=device) # cuda 0
#                 intrinsic_cur = time_cam_dataset.intrinsics[i] # cpu
#                 extrinsic_cur = time_cam_dataset.extrinsics[i] # cpu 
#                 for idx_ref, j in enumerate(sample_idxs):
#                     intrinsic_ref = time_cam_dataset.intrinsics[j]
#                     extrinsic_ref = time_cam_dataset.extrinsics[j]
#                     # The process of warping is :
#                     # 1. warp A's pixels to B (find the correspondence)
#                     # 2. use B' pixels to paint A
#                     # every coordinate of A has its corresponding coordinate in B, which size is 2
#                     warp_cur_from_ref = warp_pts_BfromA(intrinsic_cur, extrinsic_cur, key_frame_depth_maps[i], intrinsic_ref, extrinsic_ref) # (H, W, 2), cpu
#                     warp_cur_from_ref = warp_cur_from_ref.to(device) # (H, W, 2), cuda 0
#                     image_ref = sample_images_edit[idx_ref].permute(1, 2, 0).float() # cuda 0
#                     image_cur = sample_images_edit[idx_cur].permute(1, 2, 0).float() # cuda 0
#                     warp, mask = apply_warp(warp_cur_from_ref, image_cur, image_ref) # cuda 0
#                     weight = (mask != 0).sum() / (mask).numel()
#                     warp_average[mask] += warp[mask] * weight
#                     weights_mask[mask] += weight

#                 average_mask = (weights_mask != 0)
#                 warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1)
#                 sample_images_edit[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask]

#             torchvision.utils.save_image(sample_images_edit, f'output/vis/{tag}/{warm_up_idx}_sample_images_edit_after.png', nrow=sample_images_edit.shape[0], normalize=True)
#             torchvision.utils.save_image(remain_images, f'output/vis/{tag}/remain_images.png', nrow=remain_images.shape[0], normalize=True)

#             # spatial warp (based on depth map, see ViCA-NeRF)
#             remain_images_warped = remain_images.clone()
#             for idx_cur, i in enumerate(remain_idxs):
#                 warp_average = torch.zeros((H, W, 3), dtype=torch.float32, device=device) # (H, W, 3)
#                 weights_mask = torch.zeros((H, W), dtype=torch.float32, device=device) # (H, W)
#                 intrinsic_cur = time_cam_dataset.intrinsics[i] # cpu
#                 extrinsic_cur = time_cam_dataset.extrinsics[i] # cpu 
#                 for idx_ref, j in enumerate(sample_idxs):
#                     intrinsic_ref = time_cam_dataset.intrinsics[j]
#                     extrinsic_ref = time_cam_dataset.extrinsics[j]
#                     warp_cur_from_ref = warp_pts_BfromA(intrinsic_cur, extrinsic_cur, key_frame_depth_maps[i], intrinsic_ref, extrinsic_ref) # (H, W, 2), cpu
#                     warp_cur_from_ref = warp_cur_from_ref.to(device) # (H, W, 2), cuda 0
#                     image_ref = sample_images_edit[idx_ref].permute(1, 2, 0).float() # cuda 0
#                     image_cur = remain_images[idx_cur].permute(1, 2, 0).float() # cuda 0
#                     warp, mask = apply_warp(warp_cur_from_ref, image_cur, image_ref) # cuda 0
#                     weight = (mask != 0).sum() / (mask).numel()
#                     warp_average[mask] += warp[mask] * weight
#                     weights_mask[mask] += weight
                
#                 average_mask = (weights_mask != 0)
#                 warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1)
#                 remain_images_warped[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask] * warp_ratio + remain_images[idx_cur].permute(1, 2, 0)[average_mask] * (1-warp_ratio)
            
#             torchvision.utils.save_image(remain_images_warped, f'output/vis/{tag}/remain_images_warped.png', nrow=remain_images_warped.shape[0], normalize=True)

#             if warm_up_idx == warm_up_steps-1:
#                 for i in range(0, remain_images_warped.shape[0], args.sequence_length):
#                     anchor_idx = min(i, remain_images_warped.shape[0]-1)
#                     anchor_image = remain_images_warped[anchor_idx].unsqueeze(0)
#                     anchor_image_cond = remain_images_cond[anchor_idx].unsqueeze(0)

#                     start_idx = i
#                     end_idx = min(i + args.sequence_length, remain_images_warped.shape[0])
#                     selected_remain_images_warped = remain_images_warped[start_idx:end_idx] # (seq_len, 3, H, W)
#                     selected_remain_images_cond = remain_images_cond[start_idx:end_idx] # (seq_len, 3, H, W)

#                     images_input = torch.cat([anchor_image, selected_remain_images_warped], dim=0)
#                     images_cond = torch.cat([anchor_image_cond, selected_remain_images_cond], dim=0)

#                     images_edit = ip2p.edit_sequence(
#                         images=images_input.unsqueeze(0), # (1, seq_length + 1, C, H, W)
#                         images_cond=images_cond.unsqueeze(0), # (1, seq_length + 1, C, H, W)
#                         guidance_scale=args.guidance_scale,
#                         image_guidance_scale=args.image_guidance_scale,
#                         diffusion_steps=args.restview_refine_diffusion_steps,
#                         prompt=args.prompt,
#                         noisy_latent_type="noisy_latent",
#                         T=args.restview_refine_num_steps,
#                     ) # (1, C, f, H, W)

#                     images_edit = rearrange(images_edit, '1 C f H W -> f C H W').to(device, dtype=torch.float32) # (f, C, H, W)
#                     if images_edit.shape[-2:] != (H, W):
#                         images_edit = F.interpolate(images_edit, size=(H, W), mode='bilinear', align_corners=False)
                    
#                     images_edit = images_edit[1:] # (seq_len, C, H, W)
#                     remain_images_warped[start_idx:end_idx] = images_edit

#                 torchvision.utils.save_image(remain_images_warped, f'output/vis/{tag}/remain_images_warped_refined.png', nrow=remain_images_warped.shape[0], normalize=True)


#             all_rgbs.view(dataset.num_frames, num_cam, 3, H, W)[key_frame][sample_idxs] = sample_images_edit
#             all_rgbs.view(dataset.num_frames, num_cam, 3, H, W)[key_frame][remain_idxs] = remain_images_warped



#     key_frame_edit(key_frame=0, warp_ratio=0.5, warm_up_steps=12)
#     print("Key frame editing done!")

#     keyframe_images = all_rgbs[0] # [num_cam, 3, H, W]
#     torchvision.utils.save_image(keyframe_images, f'output/vis/{tag}/keyframe_images.png', nrow=args.sequence_length, normalize=True)

#     # all frame edit
#     def all_frame_update(key_frame:int=0):
#         for cam_idx in range(num_cam):

#             # edited key frame (cam_idx is the key pseudo view)
#             keyframe_image = all_rgbs[key_frame, cam_idx].unsqueeze(0) # (1, C, H, W)
#             keyframe_image_cond = original_rgbs[key_frame, cam_idx].unsqueeze(0) # (1, C, H, W)

#             for frame_idx in range(0, dataset.num_frames, args.sequence_length):
#                 start_idx = frame_idx
#                 end_idx = min(frame_idx + args.sequence_length, dataset.num_frames)
#                 selected_frame_idxs = list(range(start_idx, end_idx))
#                 sequence_length = len(selected_frame_idxs)

#                 images = all_rgbs[selected_frame_idxs, cam_idx] # (f, C, H, W), current sliding window's images
#                 images_cond = original_rgbs[selected_frame_idxs, cam_idx] # (f, C, H, W), current sliding window's condition images

#                 for i in range(0, sequence_length):
#                     if start_idx == 0 and i == 0:
#                         continue
#                     # referrence is the last frame of the previous sliding window
#                     ref_idx = max(start_idx - 1, 0)
#                     ref_image = all_rgbs[ref_idx, cam_idx].unsqueeze(0) # (1, 3, H, W)
#                     ref_image_cond = original_rgbs[ref_idx, cam_idx].unsqueeze(0) # (1, 3, H, W)

#                     cur_image = images[i].unsqueeze(0) # (1, 3, H, W)
#                     cur_image_cond = images_cond[i].unsqueeze(0) # (1, 3, H, W)

#                     ref_image = (ref_image * 255.0).float().to(args.ip2p_device)
#                     ref_image_cond = (ref_image_cond * 255.0).float().to(args.ip2p_device)
#                     cur_image = (cur_image * 255.0).float().to(args.ip2p_device)
#                     cur_image_cond = (cur_image_cond * 255.0).float().to(args.ip2p_device)

#                     padder = InputPadder(cur_image.shape)
#                     ref_image, ref_image_cond, cur_image, cur_image_cond = padder.pad(ref_image, ref_image_cond, cur_image, cur_image_cond)

#                     _, flow_fwd_ref = raft(ref_image_cond, cur_image_cond, iters=20, test_mode=True) 
#                     _, flow_bwd_ref = raft(cur_image_cond, ref_image_cond, iters=20, test_mode=True)

#                     flow_fwd_ref = padder.unpad(flow_fwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
#                     flow_bwd_ref = padder.unpad(flow_bwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 

#                     ref_image = padder.unpad(ref_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) 
#                     cur_image = padder.unpad(cur_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

#                     mask_bwd_ref = compute_bwd_mask(flow_fwd_ref, flow_bwd_ref) # (h, w)
#                     warp_cur_from_ref_proj = warp_flow(ref_image, flow_bwd_ref) # (h, w, c)

#                     warp_image = warp_cur_from_ref_proj * mask_bwd_ref[..., None] + cur_image * (1 - mask_bwd_ref[..., None]) # (h, w, c)
#                     warp_image = torch.from_numpy(warp_image / 255.0).to(images) # (h, w, c)
#                     if warp_image.shape[:2] != (H, W):
#                         warp_image = rearrange(warp_image, 'H W C -> 1 C H W')
#                         warp_image = F.interpolate(warp_image, size=(H, W), mode='bilinear',align_corners=False)
#                         warp_image = rearrange(warp_image, '1 C H W -> H W C')
#                     images[i] = warp_image.permute(2, 0, 1) # (c, h, w)
                    
#                 torchvision.utils.save_image(images, f'output/vis/{tag}/images_flow_warped.png', nrow=images.shape[0], normalize=True)
#                 torchvision.utils.save_image(images_cond, f'output/vis/{tag}/images_flow_cond.png', nrow=images_cond.shape[0], normalize=True)

#                 images = torch.cat([keyframe_image, images], dim=0) # (1 + seq_len, C, H, W)
#                 images_cond = torch.cat([keyframe_image_cond, images_cond], dim=0) # (1 + seq_len, C, H, W)

#                 images_flow = ip2p.edit_sequence(
#                     images=images.unsqueeze(0).to(args.ip2p_device), # (1, f+1, C, H, W)
#                     images_cond=images_cond.unsqueeze(0).to(args.ip2p_device), # (1, f+1, C, H, W)
#                     guidance_scale=args.guidance_scale,
#                     image_guidance_scale=args.image_guidance_scale,
#                     diffusion_steps=args.refine_diffusion_steps,
#                     prompt=args.prompt,
#                     noisy_latent_type="noisy_latent",
#                     T=args.refine_num_steps,
#                 ) # (1, C, f+1, H, W)

#                 images_flow = rearrange(images_flow, '1 C f H W -> f C H W').cpu().to(all_rgbs.dtype)
#                 images_flow = images_flow[1:] # (f, C, H, W)

#                 if images_flow.shape[-2:] != (H, W):
#                     images_flow = F.interpolate(images_flow, size=(H, W), mode='bilinear', align_corners=False)
                
#                 torchvision.utils.save_image(images_flow, f'output/vis/{tag}/images_flow_refine.png', nrow=images_flow.shape[0], normalize=True)

#                 images_flow = images_flow.to(all_rgbs)
#                 with data_lock:
#                     all_rgbs[selected_frame_idxs, cam_idx] = images_flow

#     # fork a new thread to edit all frames(time + space)
#     thread_all_frames_update = threading.Thread(target=all_frame_update, name='dataset_update')
#     thread_all_frames_update.start()
#     print('all frame update thread started')

#     pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)

#     timestamp = 0
#     for iteration in pbar:

#         with data_lock:
#             rgb_train = all_rgbs[timestamp] # [num_cam, C, H, W]

#         _, viewpoint = time_cam_dataset.next_view(timestamp)

#         for i in range(num_cam):

#             viewpoint_cam = viewpoint[i]
#             viewpoint_cam = viewpoint_cam.cuda()
            
#             rendering = render(viewpoint_cam, gaussians, pipe, background)["render"] # [C, H, W]



#             Ll1 = l1_loss(rendering, rgb_train[i])
#             Lssim = 1.0 - ssim(rendering, rgb_train[i])
#             loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

#             loss.backward()

#             gaussians.optimizer.step()
#             gaussians.optimizer.zero_grad(set_to_none = True)
#             if pipe.env_map_res and iteration < pipe.env_optimize_until:
#                 env_map_optimizer.step()
#                 env_map_optimizer.zero_grad(set_to_none = True)

#         timestamp += 1
#         if timestamp == dataset.num_frames:
#             timestamp = 0
        
#         if (iteration in args.save_iterations):
#             print("\n[ITER {}] Saving Gaussians".format(iteration))
#             scene.save_edit(iteration)
    
#         with torch.no_grad():
#             gaussians.optimizer.step()
#             gaussians.optimizer.zero_grad(set_to_none = True)
#             if pipe.env_map_res and iteration < pipe.env_optimize_until:
#                 env_map_optimizer.step()
#                 env_map_optimizer.zero_grad(set_to_none = True)

#             torchvision.utils.save_image(rendering, f'output/vis/{tag}/gaussian_editted.png', nrow=1, normalize=True)


# if __name__ == '__main__':
#     # Set up argument parser and merge config file(yaml)
#     parser = ArgumentParser(description='Render script parameters')
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#     parser.add_argument("--config", type=str, default="configs/dynerf/coffee_martini_edit.yaml")
#     parser.add_argument("--chkpnt_path", type=str, default="output/N3V/coffee_martini/chkpnt_best.pth")
#     parser.add_argument("--seed", type=int, default=6666)

#     parser.add_argument("--progress_refresh_rate", type=int, default=10,
#                         help='how many iterations to show psnrs or iters')
    
#     # parser.add_argument("--vis")
    
#     # TODO
#     parser.add_argument("--save_iterations", nargs="+", type=int, default=[70, 200, 290])

#     # loader options
#     parser.add_argument("--batch_size", type=int, default=4096)
#     parser.add_argument("--patch_size", type=int, default=32)
#     # TODO
#     parser.add_argument("--n_iters", type=int, default=300) # 30000
#     parser.add_argument("--n_keyframe_iters", type=int, default=800)
#     parser.add_argument('--dataset_name', type=str, default='blender', choices=['n3dv_dynamic','deepview_dynamic',])

#     # edit params
#     parser.add_argument("--editted_path", type=str, default="output/editted_N3V/coffee_martini")
#     # sequence_ip2p options
#     parser.add_argument('--prompt', type=str, default="What if it was painted by Van Gogh?",
#                         help='prompt for InstructPix2Pix')
#     parser.add_argument('--guidance_scale', type=float, default=7.5,
#                         help='(text) guidance scale for InstructPix2Pix')
#     parser.add_argument('--image_guidance_scale', type=float, default=1.5,
#                         help='image guidance scale for InstructPix2Pix')
#     parser.add_argument('--keyframe_refine_diffusion_steps', type=int, default=10,
#                         help='number of diffusion steps to take for keyframe refinement')
#     parser.add_argument('--keyframe_refine_num_steps', type=int, default=700,
#                         help='number of denoise steps to take for keyframe refinement')
#     parser.add_argument('--restview_refine_diffusion_steps', type=int, default=10,
#                         help='number of diffusion steps to take for keyframe refinement')
#     parser.add_argument('--restview_refine_num_steps', type=int, default=800,
#                         help='number of denoise steps to take for keyframe refinement')
#     parser.add_argument('--diffusion_steps', type=int, default=20,
#                         help='number of diffusion steps to take for InstructPix2Pix')
#     parser.add_argument('--refine_diffusion_steps', type=int, default=6,
#                         help='number of diffusion steps to take for refinement')
#     parser.add_argument('--refine_num_steps', type=int, default=700,
#                         help='number of denoise steps to take for refinement')
#     parser.add_argument('--ip2p_device', type=str, default='cuda:1',
#                         help='second device to place InstructPix2Pix on')
#     parser.add_argument('--ip2p_use_full_precision', type=bool, default=False,
#                         help='Whether to use full precision for InstructPix2Pix')
#     parser.add_argument('--sequence_length', type=int, default=5,
#                         help='length of the sequence')
#     parser.add_argument('--overlap_length', type=int, default=1,
#                         help='length of the overlap')
#     parser.add_argument('--raft_ckpt', type=str, default='./ip2p_models/weights/raft-things.pth',)

#     parser.add_argument('--datadir', type=str, default="")
#     parser.add_argument('--basedir', type=str, default="")
#     parser.add_argument('--expname', type=str, default="")
#     parser.add_argument('--cache', type=str, default="")
    
#     args = parser.parse_args()

#     cfg = OmegaConf.load(args.config)

#     def recursive_merge(key, host):
#         global dataset, opt, pipe
#         if isinstance(host[key], DictConfig):
#             for key1 in host[key].keys():
#                 recursive_merge(key1, host[key])
#         else:
#             # assert hasattr(args, key), key
#             setattr(args, key, host[key])
#     for k in cfg.keys():
#         recursive_merge(k, cfg)

#     setup_seed(args.seed)

#     dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

#     gaussian_editing(args, dataset, opt, pipe)

import os
from os import makedirs
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim, msssim
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import numpy as np
import random
from tqdm import tqdm
from time import time
from gaussian_renderer import render
import concurrent.futures
import imageio
from ip2p_sequence import SequenceInstructPix2Pix
from ip2p_models.RAFT.raft import RAFT
from ip2p_models.RAFT.utils.utils import InputPadder
from einops import rearrange
import datetime
import threading
import math
import sys
import cv2
import ipdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)

enhomo0 = lambda x: torch.cat([x, torch.ones([1, *x.shape[1:]], device=x.device, dtype=x.dtype)], dim=0)
dehomo0 = lambda x: x[:-1] / x[-1:]

def warp_pts_BfromA(intriA, w2cA, depthA, intriB, w2cB):
    """
    Warp pixels from view A to view B.

    Args:
        intriA: Intrinsic matrix of view A (3, 3).
        w2cA: World to camera transformation matrix of view A (4, 4).
        depthA: Depth map of view A (H, W).
        intriB: Intrinsic matrix of view B (3, 3).
        w2cB: World to camera transformation matrix of view B (4, 4).

    Returns:
        Corresponding pixel coordinates in view B.
    """
    # Get the height and width from the depth map
    H, W = depthA.shape

    # Create a meshgrid for the pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(H, device=depthA.device, dtype=torch.float32), 
        torch.arange(W, device=depthA.device, dtype=torch.float32), 
        indexing='ij'
    )
    x = x.flatten()
    y = y.flatten()

    # Create pixel coordinates in homogeneous form (3, N)
    pix_coords_A = torch.stack([x, y, torch.ones_like(x, device=depthA.device)], dim=0)

    # Convert pixel coordinates to camera coordinates (3, N)
    cam_coords_A = torch.matmul(torch.inverse(intriA), pix_coords_A) * depthA.flatten()

    # Convert camera coordinates to world coordinates (4, N)
    cam_coords_A_homo = enhomo0(cam_coords_A)
    world_coords_A = torch.matmul(torch.inverse(w2cA), cam_coords_A_homo)

    # Convert world coordinates to camera coordinates in view B (4, N)
    cam_coords_B_homo = torch.matmul(w2cB, world_coords_A)

    # Convert camera coordinates to pixel coordinates in view B (3, N)
    pix_coords_B_homo = torch.matmul(intriB, dehomo0(cam_coords_B_homo))

    # Normalize homogeneous coordinates to get final pixel coordinates in view B
    pix_coords_B = dehomo0(pix_coords_B_homo).reshape(2, H, W)

    return pix_coords_B.permute(1, 2, 0)


def apply_warp(
    AfromB: torch.Tensor, 
    imgA: torch.Tensor, 
    imgB: torch.Tensor,
    u=None, d=None, l=None, r=None, 
    default=torch.tensor([0, 0, 0]),
    threshold=0.01
):
    """
    Warp imgB to imgA based on precomputed correspondence AfromB.
    
    Args:
        AfromB: Precomputed correspondence from B to A, shape (H, W, 2).
        imgA: Target image A, shape (H, W, C).
        imgB: Source image B, shape (H, W, C).
        u, d, l, r: Bounds for cropping the image.
        default: Default color for out-of-boundary pixels or mismatches.
        threshold: Threshold to filter out less accurate warps.
    
    Returns:
        imgA_warped: Warped image with pixels from imgB.
        mask: Boolean mask indicating valid pixels.
    """
    # Set boundaries
    u, d, l, r = u or 0, d or imgA.shape[0], l or 0, r or imgA.shape[1]
    default = default.to(dtype=imgA.dtype, device=imgA.device)
    
    # Extract X and Y coordinates from AfromB
    Y, X = AfromB.permute(2, 0, 1)
    
    # Calculate the mask for valid coordinates within bounds
    mask = (u <= X) & (X < d) & (l <= Y) & (Y < r)
    
    # 引入一个基于阈值的附加过滤条件，过滤掉不准确的映射点
    delta = torch.sqrt((X - torch.round(X)) ** 2 + (Y - torch.round(Y)) ** 2)
    mask &= delta < threshold
    
    # Normalize the coordinates to [-1, 1] for grid_sample
    X_norm = ((X - u) / (d - 1 - u) * 2 - 1) * mask
    Y_norm = ((Y - l) / (r - 1 - l) * 2 - 1) * mask
    pix = torch.stack([Y_norm, X_norm], dim=-1).unsqueeze(0)  # shape (1, H, W, 2)
    
    # Warp imgB to imgA using grid_sample
    imgA_warped = F.grid_sample(imgB.permute(2, 0, 1).unsqueeze(0), pix, mode='bilinear', align_corners=True)
    imgA_warped = imgA_warped.squeeze(0).permute(1, 2, 0)
    
    # Apply the mask to imgA_warped, setting invalid pixels to default color
    imgA_warped[~mask] = default
    
    return imgA_warped, mask


def warp_flow(img, flow): 
    # warp image according to flow
    h, w = flow.shape[:2]
    flow_new = flow.copy() 
    flow_new[:, :, 0] += np.arange(w) 
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis] 

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def compute_bwd_mask(fwd_flow, bwd_flow):
    # compute the backward mask
    alpha_1 = 0.5 
    alpha_2 = 0.5

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return bwd_mask

    

def gaussian_editing(args, dataset, opt, pipe):
    # log
    tag_prompt = args.prompt.split(' ')[-1].replace('?','')
    tag = f'{datetime.datetime.now().strftime("%d-%H-%M")}_{tag_prompt}'


    # load gaussians and scene
    gaussians = GaussianModel(args.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if args.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, shuffle=False, num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
    (model_param, _) = torch.load(args.chkpnt_path)
    gaussians.restore(model_param, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3, pipe.env_map_res, pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(False))
        env_map_optimizer = torch.optim.Adam([env_map], lr=opt.feature_lr, eps=1e-15)
    else:
        env_map = None

    gaussians.env_map = env_map

    num_cam = len([f for f in os.listdir(args.source_path) if f.endswith('.mp4')])

    # load ip2p, raft
    ip2p = SequenceInstructPix2Pix(device=args.ip2p_device, ip2p_use_full_precision=args.ip2p_use_full_precision)
    
    raft = torch.nn.DataParallel(RAFT(args=Namespace(small=False, mixed_precision=False)))
    raft.load_state_dict(torch.load(args.raft_ckpt))
    raft = raft.module
    
    raft = raft.to(args.ip2p_device)
    raft.requires_grad_(False)
    raft.eval()
    print('RAFT loaded!')

    # Sampler, 根据训练的iteration来选择不同的Sampler
    # trainingSampler = MotionSampler(allrgbs, args.num_frames, args.batch_size)
    # simpleSampler = SimpleSampler(args.num_frames, args.batch_size)

    
    time_cam_dataset = scene.getTrainCameras()  # all_rgbs


    # cache for depth
    cache_depth_path = os.path.join(args.cache_path, args.exp_name, 'depth_all.pt')
    if os.path.exists(cache_depth_path):
        key_frame_depth_maps = torch.load(cache_depth_path).cpu()
        print('all depth maps loaded from cache dir')
        torch.cuda.empty_cache()
    else:
        cache_path = os.path.join(args.cache_path, args.exp_name)
        os.makedirs(cache_path, exist_ok=True)
        key_frame_depth_maps = []
        key_frame_dataset = scene.getKeyCameras()

        with torch.no_grad():
            for idx, view in enumerate(tqdm(key_frame_dataset, desc="Rendering depth maps for key frames")):
                _, viewpoint_cam = view
                viewpoint_cam = viewpoint_cam.cuda()

                depth_map = render(viewpoint_cam, gaussians, pipe, background)["depth"] # (1, H, W)
                key_frame_depth_maps.append(depth_map.squeeze(0))
            key_frame_depth_maps = torch.stack(key_frame_depth_maps, dim=0) # [cam_num, H, W]
            torch.save(key_frame_depth_maps, cache_depth_path)
            print("all key frame's depth maps saved to cache dir")
            torch.cuda.empty_cache()
    
    all_rgbs = time_cam_dataset.time_cam_image.to(device) # [num_frame * num_cam, 3, H, W], cuda 0
    H, W = all_rgbs.shape[-2:]
    all_rgbs = all_rgbs.view(num_cam, dataset.num_frames, 3, H, W).permute(1, 0, 2, 3, 4) # [num_frame, num_cam, 3, H, W]
    original_rgbs = all_rgbs.clone() # [num_frame, num_cam, 3, H, W]
    cam_idxs = list(range(0, num_cam))

    # thread lock for data and model
    data_lock = threading.Lock()

    os.makedirs(f'output/vis/{tag}', exist_ok=True)

    def key_frame_edit(key_frame:int = 0, warp_ratio:float = 0.5, warm_up_steps:int = 12):
        print(f'key frame {key_frame} editing')
        for warm_up_idx in range(warm_up_steps):
            sample_idxs = sorted(list(np.random.choice(cam_idxs, args.sequence_length, replace=False)))
            remain_idxs = sorted(list(set(cam_idxs) - set(sample_idxs)))

            sample_images = all_rgbs[key_frame][sample_idxs] # [sample_length, 3, H, W], cuda 0
            sample_images_cond = original_rgbs[key_frame][sample_idxs] # [sample_length, 3, H, W], cuda 0

            remain_images = all_rgbs[key_frame][remain_idxs] # [remain_length, 3, H, W], cuda 0
            remain_images_cond = original_rgbs[key_frame][remain_idxs] # [remain_length, 3, H, W], cuda 0

            torchvision.utils.save_image(sample_images, f'output/vis/{tag}/sample_images.png', nrow=sample_images.shape[0], normalize=True)
            torchvision.utils.save_image(sample_images_cond, f'output/vis/{tag}/sample_images_cond.png', nrow=sample_images_cond.shape[0], normalize=True)

            # cosine annealing
            scale_min, scale_max = 0.8, 1.0
            scale = scale_min + 0.5 * (1 + math.cos(math.pi * warm_up_idx / warm_up_steps)) * (scale_max - scale_min)
            sample_images_edit = ip2p.edit_sequence(
                images=sample_images.unsqueeze(0), # (1, sample_length, C, H, W)
                images_cond=sample_images_cond.unsqueeze(0), # (1, sample_length, C, H, W)
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                diffusion_steps=int(args.diffusion_steps * scale),
                prompt=args.prompt,
                noisy_latent_type="noisy_latent",
                T=int(1000 * scale),
            ) # (1, C, f, H, W), cuda 0

            sample_images_edit = rearrange(sample_images_edit, '1 C f H W -> f C H W').to(device, dtype=torch.float32) # (f, C, H, W), cuda 0
            if sample_images_edit.shape[-2:] != (H, W):
                sample_images_edit = F.interpolate(sample_images_edit, size=(H, W), mode='bilinear', align_corners=False)

            torchvision.utils.save_image(sample_images_edit, f'output/vis/{tag}/{warm_up_idx}_sample_images_edit_before.png', nrow=sample_images_edit.shape[0], normalize=True)

            # spatial warp (based on depth map, see ViCA-NeRF)
            for idx_cur, i in enumerate(sample_idxs):
                warp_average = torch.zeros((H, W, 3), dtype=torch.float32, device=device) # cuda 0
                weights_mask = torch.zeros((H, W), dtype=torch.float32, device=device) # cuda 0
                intrinsic_cur = time_cam_dataset.intrinsics[i] # cpu
                extrinsic_cur = time_cam_dataset.extrinsics[i] # cpu 
                for idx_ref, j in enumerate(sample_idxs):
                    intrinsic_ref = time_cam_dataset.intrinsics[j]
                    extrinsic_ref = time_cam_dataset.extrinsics[j]
                    # The process of warping is :
                    # 1. warp A's pixels to B (find the correspondence)
                    # 2. use B' pixels to paint A
                    # every coordinate of A has its corresponding coordinate in B, which size is 2
                    warp_cur_from_ref = warp_pts_BfromA(intrinsic_cur, extrinsic_cur, key_frame_depth_maps[i], intrinsic_ref, extrinsic_ref) # (H, W, 2), cpu
                    warp_cur_from_ref = warp_cur_from_ref.to(device) # (H, W, 2), cuda 0
                    image_ref = sample_images_edit[idx_ref].permute(1, 2, 0).float() # cuda 0
                    image_cur = sample_images_edit[idx_cur].permute(1, 2, 0).float() # cuda 0
                    warp, mask = apply_warp(warp_cur_from_ref, image_cur, image_ref) # cuda 0
                    weight = (mask != 0).sum() / (mask).numel()
                    warp_average[mask] += warp[mask] * weight
                    weights_mask[mask] += weight

                average_mask = (weights_mask != 0)
                warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1)
                sample_images_edit[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask]

            torchvision.utils.save_image(sample_images_edit, f'output/vis/{tag}/{warm_up_idx}_sample_images_edit_after.png', nrow=sample_images_edit.shape[0], normalize=True)
            torchvision.utils.save_image(remain_images, f'output/vis/{tag}/remain_images.png', nrow=remain_images.shape[0], normalize=True)

            # spatial warp (based on depth map, see ViCA-NeRF)
            remain_images_warped = remain_images.clone()
            for idx_cur, i in enumerate(remain_idxs):
                warp_average = torch.zeros((H, W, 3), dtype=torch.float32, device=device) # (H, W, 3)
                weights_mask = torch.zeros((H, W), dtype=torch.float32, device=device) # (H, W)
                intrinsic_cur = time_cam_dataset.intrinsics[i] # cpu
                extrinsic_cur = time_cam_dataset.extrinsics[i] # cpu 
                for idx_ref, j in enumerate(sample_idxs):
                    intrinsic_ref = time_cam_dataset.intrinsics[j]
                    extrinsic_ref = time_cam_dataset.extrinsics[j]
                    warp_cur_from_ref = warp_pts_BfromA(intrinsic_cur, extrinsic_cur, key_frame_depth_maps[i], intrinsic_ref, extrinsic_ref) # (H, W, 2), cpu
                    warp_cur_from_ref = warp_cur_from_ref.to(device) # (H, W, 2), cuda 0
                    image_ref = sample_images_edit[idx_ref].permute(1, 2, 0).float() # cuda 0
                    image_cur = remain_images[idx_cur].permute(1, 2, 0).float() # cuda 0
                    warp, mask = apply_warp(warp_cur_from_ref, image_cur, image_ref) # cuda 0
                    weight = (mask != 0).sum() / (mask).numel()
                    warp_average[mask] += warp[mask] * weight
                    weights_mask[mask] += weight
                
                average_mask = (weights_mask != 0)
                warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1)
                remain_images_warped[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask] * warp_ratio + remain_images[idx_cur].permute(1, 2, 0)[average_mask] * (1-warp_ratio)
            
            torchvision.utils.save_image(remain_images_warped, f'output/vis/{tag}/remain_images_warped.png', nrow=remain_images_warped.shape[0], normalize=True)

            if warm_up_idx == warm_up_steps-1:
                for i in range(0, remain_images_warped.shape[0], args.sequence_length):
                    anchor_idx = min(i, remain_images_warped.shape[0]-1)
                    anchor_image = remain_images_warped[anchor_idx].unsqueeze(0)
                    anchor_image_cond = remain_images_cond[anchor_idx].unsqueeze(0)

                    start_idx = i
                    end_idx = min(i + args.sequence_length, remain_images_warped.shape[0])
                    selected_remain_images_warped = remain_images_warped[start_idx:end_idx] # (seq_len, 3, H, W)
                    selected_remain_images_cond = remain_images_cond[start_idx:end_idx] # (seq_len, 3, H, W)

                    images_input = torch.cat([anchor_image, selected_remain_images_warped], dim=0)
                    images_cond = torch.cat([anchor_image_cond, selected_remain_images_cond], dim=0)

                    images_edit = ip2p.edit_sequence(
                        images=images_input.unsqueeze(0), # (1, seq_length + 1, C, H, W)
                        images_cond=images_cond.unsqueeze(0), # (1, seq_length + 1, C, H, W)
                        guidance_scale=args.guidance_scale,
                        image_guidance_scale=args.image_guidance_scale,
                        diffusion_steps=args.restview_refine_diffusion_steps,
                        prompt=args.prompt,
                        noisy_latent_type="noisy_latent",
                        T=args.restview_refine_num_steps,
                    ) # (1, C, f, H, W)

                    images_edit = rearrange(images_edit, '1 C f H W -> f C H W').to(device, dtype=torch.float32) # (f, C, H, W)
                    if images_edit.shape[-2:] != (H, W):
                        images_edit = F.interpolate(images_edit, size=(H, W), mode='bilinear', align_corners=False)
                    
                    images_edit = images_edit[1:] # (seq_len, C, H, W)
                    remain_images_warped[start_idx:end_idx] = images_edit

                torchvision.utils.save_image(remain_images_warped, f'output/vis/{tag}/remain_images_warped_refined.png', nrow=remain_images_warped.shape[0], normalize=True)


            all_rgbs.view(dataset.num_frames, num_cam, 3, H, W)[key_frame][sample_idxs] = sample_images_edit
            all_rgbs.view(dataset.num_frames, num_cam, 3, H, W)[key_frame][remain_idxs] = remain_images_warped



    key_frame_edit(key_frame=0, warp_ratio=0.5, warm_up_steps=12)
    print("Key frame editing done!")

    keyframe_images = all_rgbs[0] # [num_cam, 3, H, W]
    torchvision.utils.save_image(keyframe_images, f'output/vis/{tag}/keyframe_images.png', nrow=args.sequence_length, normalize=True)

    # all frame edit
    def all_frame_update(key_frame:int=0):
        for cam_idx in range(num_cam):

            # edited key frame (cam_idx is the key pseudo view)
            keyframe_image = all_rgbs[key_frame, cam_idx].unsqueeze(0) # (1, C, H, W)
            keyframe_image_cond = original_rgbs[key_frame, cam_idx].unsqueeze(0) # (1, C, H, W)

            for frame_idx in range(0, dataset.num_frames, args.sequence_length):
                start_idx = frame_idx
                end_idx = min(frame_idx + args.sequence_length, dataset.num_frames)
                selected_frame_idxs = list(range(start_idx, end_idx))
                sequence_length = len(selected_frame_idxs)

                images = all_rgbs[selected_frame_idxs, cam_idx] # (f, C, H, W), current sliding window's images
                images_cond = original_rgbs[selected_frame_idxs, cam_idx] # (f, C, H, W), current sliding window's condition images

                for i in range(0, sequence_length):
                    if start_idx == 0 and i == 0:
                        continue
                    # referrence is the last frame of the previous sliding window
                    ref_idx = max(start_idx - 1, 0)
                    ref_image = all_rgbs[ref_idx, cam_idx].unsqueeze(0) # (1, 3, H, W)
                    ref_image_cond = original_rgbs[ref_idx, cam_idx].unsqueeze(0) # (1, 3, H, W)

                    cur_image = images[i].unsqueeze(0) # (1, 3, H, W)
                    cur_image_cond = images_cond[i].unsqueeze(0) # (1, 3, H, W)

                    ref_image = (ref_image * 255.0).float().to(args.ip2p_device)
                    ref_image_cond = (ref_image_cond * 255.0).float().to(args.ip2p_device)
                    cur_image = (cur_image * 255.0).float().to(args.ip2p_device)
                    cur_image_cond = (cur_image_cond * 255.0).float().to(args.ip2p_device)

                    padder = InputPadder(cur_image.shape)
                    ref_image, ref_image_cond, cur_image, cur_image_cond = padder.pad(ref_image, ref_image_cond, cur_image, cur_image_cond)

                    _, flow_fwd_ref = raft(ref_image_cond, cur_image_cond, iters=20, test_mode=True) 
                    _, flow_bwd_ref = raft(cur_image_cond, ref_image_cond, iters=20, test_mode=True)

                    flow_fwd_ref = padder.unpad(flow_fwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
                    flow_bwd_ref = padder.unpad(flow_bwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 

                    ref_image = padder.unpad(ref_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) 
                    cur_image = padder.unpad(cur_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

                    mask_bwd_ref = compute_bwd_mask(flow_fwd_ref, flow_bwd_ref) # (h, w)
                    warp_cur_from_ref_proj = warp_flow(ref_image, flow_bwd_ref) # (h, w, c)

                    warp_image = warp_cur_from_ref_proj * mask_bwd_ref[..., None] + cur_image * (1 - mask_bwd_ref[..., None]) # (h, w, c)
                    warp_image = torch.from_numpy(warp_image / 255.0).to(images) # (h, w, c)
                    if warp_image.shape[:2] != (H, W):
                        warp_image = rearrange(warp_image, 'H W C -> 1 C H W')
                        warp_image = F.interpolate(warp_image, size=(H, W), mode='bilinear',align_corners=False)
                        warp_image = rearrange(warp_image, '1 C H W -> H W C')
                    images[i] = warp_image.permute(2, 0, 1) # (c, h, w)
                    
                torchvision.utils.save_image(images, f'output/vis/{tag}/images_flow_warped.png', nrow=images.shape[0], normalize=True)
                torchvision.utils.save_image(images_cond, f'output/vis/{tag}/images_flow_cond.png', nrow=images_cond.shape[0], normalize=True)

                images = torch.cat([keyframe_image, images], dim=0) # (1 + seq_len, C, H, W)
                images_cond = torch.cat([keyframe_image_cond, images_cond], dim=0) # (1 + seq_len, C, H, W)

                images_flow = ip2p.edit_sequence(
                    images=images.unsqueeze(0).to(args.ip2p_device), # (1, f+1, C, H, W)
                    images_cond=images_cond.unsqueeze(0).to(args.ip2p_device), # (1, f+1, C, H, W)
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    diffusion_steps=args.refine_diffusion_steps,
                    prompt=args.prompt,
                    noisy_latent_type="noisy_latent",
                    T=args.refine_num_steps,
                ) # (1, C, f+1, H, W)

                images_flow = rearrange(images_flow, '1 C f H W -> f C H W').cpu().to(all_rgbs.dtype)
                images_flow = images_flow[1:] # (f, C, H, W)

                if images_flow.shape[-2:] != (H, W):
                    images_flow = F.interpolate(images_flow, size=(H, W), mode='bilinear', align_corners=False)
                
                torchvision.utils.save_image(images_flow, f'output/vis/{tag}/images_flow_refine.png', nrow=images_flow.shape[0], normalize=True)

                images_flow = images_flow.to(all_rgbs)
                with data_lock:
                    all_rgbs[selected_frame_idxs, cam_idx] = images_flow

    # fork a new thread to edit all frames(time + space)
    thread_all_frames_update = threading.Thread(target=all_frame_update, name='dataset_update')
    thread_all_frames_update.start()
    print('all frame update thread started')

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)

    timestamp = 0
    for iteration in pbar:

        with data_lock:
            rgb_train = all_rgbs[timestamp] # [num_cam, C, H, W]

        _, viewpoint = time_cam_dataset.next_view(timestamp)

        for i in range(num_cam):

            viewpoint_cam = viewpoint[i]
            viewpoint_cam = viewpoint_cam.cuda()
            
            rendering = render(viewpoint_cam, gaussians, pipe, background)["render"] # [C, H, W]



            Ll1 = l1_loss(rendering, rgb_train[i])
            Lssim = 1.0 - ssim(rendering, rgb_train[i])
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

            loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if pipe.env_map_res and iteration < pipe.env_optimize_until:
                env_map_optimizer.step()
                env_map_optimizer.zero_grad(set_to_none = True)

        timestamp += 1
        if timestamp == dataset.num_frames:
            timestamp = 0
        
        if (iteration in args.save_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save_edit(iteration)
    
        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if pipe.env_map_res and iteration < pipe.env_optimize_until:
                env_map_optimizer.step()
                env_map_optimizer.zero_grad(set_to_none = True)

            torchvision.utils.save_image(rendering, f'output/vis/{tag}/gaussian_editted.png', nrow=1, normalize=True)


if __name__ == '__main__':
    # Set up argument parser and merge config file(yaml)
    parser = ArgumentParser(description='Render script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="configs/dynerf/coffee_martini_edit.yaml")
    parser.add_argument("--chkpnt_path", type=str, default="output/N3V/coffee_martini/chkpnt_best.pth")
    parser.add_argument("--seed", type=int, default=6666)

    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')
    
    # parser.add_argument("--vis")
    
    # TODO
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[70, 200, 290])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--patch_size", type=int, default=32)
    # TODO
    parser.add_argument("--n_iters", type=int, default=300) # 30000
    parser.add_argument("--n_keyframe_iters", type=int, default=800)
    parser.add_argument('--dataset_name', type=str, default='blender', choices=['n3dv_dynamic','deepview_dynamic',])

    # edit params
    parser.add_argument("--editted_path", type=str, default="output/editted_N3V/coffee_martini")
    # sequence_ip2p options
    parser.add_argument('--prompt', type=str, default="What if it was painted by Van Gogh?",
                        help='prompt for InstructPix2Pix')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='(text) guidance scale for InstructPix2Pix')
    parser.add_argument('--image_guidance_scale', type=float, default=1.5,
                        help='image guidance scale for InstructPix2Pix')
    parser.add_argument('--keyframe_refine_diffusion_steps', type=int, default=10,
                        help='number of diffusion steps to take for keyframe refinement')
    parser.add_argument('--keyframe_refine_num_steps', type=int, default=700,
                        help='number of denoise steps to take for keyframe refinement')
    parser.add_argument('--restview_refine_diffusion_steps', type=int, default=10,
                        help='number of diffusion steps to take for keyframe refinement')
    parser.add_argument('--restview_refine_num_steps', type=int, default=800,
                        help='number of denoise steps to take for keyframe refinement')
    parser.add_argument('--diffusion_steps', type=int, default=20,
                        help='number of diffusion steps to take for InstructPix2Pix')
    parser.add_argument('--refine_diffusion_steps', type=int, default=6,
                        help='number of diffusion steps to take for refinement')
    parser.add_argument('--refine_num_steps', type=int, default=700,
                        help='number of denoise steps to take for refinement')
    parser.add_argument('--ip2p_device', type=str, default='cuda:1',
                        help='second device to place InstructPix2Pix on')
    parser.add_argument('--ip2p_use_full_precision', type=bool, default=False,
                        help='Whether to use full precision for InstructPix2Pix')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='length of the sequence')
    parser.add_argument('--overlap_length', type=int, default=1,
                        help='length of the overlap')
    parser.add_argument('--raft_ckpt', type=str, default='./ip2p_models/weights/raft-things.pth',)

    parser.add_argument('--datadir', type=str, default="")
    parser.add_argument('--basedir', type=str, default="")
    parser.add_argument('--expname', type=str, default="")
    parser.add_argument('--cache', type=str, default="")
    
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        global dataset, opt, pipe
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            # assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)

    setup_seed(args.seed)

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    gaussian_editing(args, dataset, opt, pipe)