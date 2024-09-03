import os
from os import makedirs
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
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
    y, x = torch.meshgrid(torch.arange(H, device=depthA.device), torch.arange(W, device=depthA.device))
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
    default=torch.tensor([0, 0, 0]) 
):
    """
    Warp imgB to imgA based on precomputed correspondence AfromB.
    
    Args:
        AfromB: Precomputed correspondence from B to A, shape (H, W, 2).
        imgA: Target image A, shape (H, W, C).
        imgB: Source image B, shape (H, W, C).
        u, d, l, r: Bounds for cropping the image.
        default: Default color for out-of-boundary pixels or mismatches.
    
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
    
    # Normalize the coordinates to [-1, 1] for grid_sample
    X_norm = ((X - u) / (d - 1 - u) * 2 - 1) * mask
    Y_norm = ((Y - l) / (r - 1 - l) * 2 - 1) * mask
    pix = torch.stack([-Y_norm, X_norm], dim=-1).unsqueeze(0)  # shape (1, H, W, 2)
    
    # Warp imgB to imgA using grid_sample
    imgA_warped = F.grid_sample(imgB.permute(2, 0, 1).unsqueeze(0), pix, mode='bilinear', align_corners=True)
    imgA_warped = imgA_warped.squeeze(0).permute(1, 2, 0)
    
    # Apply the mask to imgA_warped, setting invalid pixels to default color
    imgA_warped[~mask] = default
    
    return imgA_warped, mask



# 简单 Sampler，随机采样
class SimpleSampler:
    def __init__(self, total_frame, batch):
        self.total_frame = total_frame
        self.batch = batch
        self.permute_base = self.gen_permute()

    def nextids(self):
        frame = int(random.random()*self.total_frame)
        start = int(random.random()*(len(self.permute_base)-self.batch))
        return self.permute_base[start:start+self.batch]+frame*self.per_frame_length

    def gen_permute(self):
        self.per_frame_length = self.total / self.total_frame
        assert self.per_frame_length.is_integer()
        self.per_frame_length = int(self.per_frame_length)
        return torch.LongTensor(np.random.permutation(self.per_frame_length))

# 基于动作复杂度（幅度）的 Sampler
# class MotionSampler:
#     def __init__(self, allrgbs, total_frame, batch):
#         self.total = allrgbs.shape[0]
#         self.total_frame = total_frame
#         self.batch = batch      # batch_size
#         self.curr = self.total
#         self.ids = None
#         self.per_frame_length = self.total / self.total_frame 
#         assert self.per_frame_length.is_integer()
#         self.per_frame_length = int(self.per_frame_length) 
#         self.permute_base = torch.LongTensor(np.random.permutation(self.per_frame_length)) 
#         motion_mask = (allrgbs-torch.roll(allrgbs,self.per_frame_length,0)).abs().mean(-1)>(10/255) 
#         get_mask = lambda x: motion_mask[x*self.per_frame_length:(x+1)*self.per_frame_length] 
#         self.mi = {} # motion index
#         for k in range(self.total_frame):
#             # nearby 5 frames
#             current_mask = get_mask(k) 
#             for i in range(1,6): 
#                 if k-i >= 0: 
#                     current_mask = current_mask|get_mask(k-i) 
#                 if k+i < self.total_frame: 
#                     current_mask = current_mask|get_mask(k+i) 
#             mask_idx = current_mask.nonzero()
#             if len(mask_idx)>0: 
#                 self.mi[k] = mask_idx[:,0] 
#             else:
#                 self.mi[k] = []
#         self.motion_num = self.batch//10

#     def nextids(self):
#         # self.curr+=self.batch
#         # if self.curr + self.batch > self.total:
#         #     self.ids = self.gen_permute()
#         #     self.curr = 0
#         # return self.ids[self.curr:self.curr+self.batch]
#         frame = int(random.random()*self.total_frame)
#         start = int(random.random()*(len(self.permute_base)))
#         m_num = len(self.mi[frame])
#         if m_num > 0:
#             if m_num < self.motion_num:
#                 m_idx = self.mi[frame][torch.randperm(self.motion_num)%m_num]
#             else:
#                 m_idx = self.mi[frame][torch.randperm(m_num)[:self.motion_num]]
#         else:
#             m_idx = self.permute_base[:1]
#         # start = int(random.random()*(len(self.permute_base)-self.batch+len(m_idx)))
#         end = min(start+self.batch-len(m_idx), self.per_frame_length)
#         return  torch.cat([m_idx, self.permute_base[start:end]],0)+frame*self.per_frame_length

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
        key_frame_depth_maps = []
        # TODO：这里 getKeyCameras 尚未实现，应该是 key frame，我看完edit key frame再决定
        key_frame_dataset = scene.getKeyCameras()

        with torch.no_grad():
            for idx, view in enumerate(tqdm(key_frame_dataset, desc="Rendering depth maps for key frames")):
                _, viewpoint_cam = view
                viewpoint_cam = viewpoint_cam.cuda()

                depth_map = render(viewpoint_cam, gaussians, pipe, background)["depth"] # (1, H, W)
                depth_map = depth_map.permute(1, 2, 0).view(-1, 1).cpu() # (H * W, 1)
                key_frame_depth_maps.append(depth_map)
            key_frame_depth_maps = torch.stack(key_frame_depth_maps, dim=0) # [cam_num, H * W, 1]
            torch.save(key_frame_depth_maps, cache_depth_path)
            print("all key frame's depth maps saved to cache dir")
            torch.cuda.empty_cache()
    
    all_rgbs = time_cam_dataset.time_cam_image # [num_frame, num_cam, 3, H, W]
    H, W = all_rgbs.shape[-2:]
    original_rgbs = time_cam_dataset.time_cam_image.clone() # [num_frame, num_cam, 3, H, W]
    cam_idxs = list(range(0, num_cam))

    # thread lock for data and model
    data_lock = threading.Lock()
    model_lock = threading.Lock()

    def key_frame_edit(key_frame:int = 0, warp_ratio:float = 0.5, warm_up_steps:int = 12):
        print(f'key frame {key_frame} editing')
        for warm_up_idx in range(warm_up_steps):
            sample_idxs = sorted(list(np.random.choice(cam_idxs, args.sequence_length, replace=False)))
            remain_idxs = sorted(list(set(cam_idxs) - set(sample_idxs)))

            sample_images = all_rgbs[key_frame][sample_idxs].to(device) # [sample_length, 3, H, W]
            sample_images_cond = original_rgbs[key_frame][sample_idxs].to(device) # [sample_length, 3, H, W]

            remain_images = all_rgbs[key_frame][remain_idxs].to(device) # [remain_length, 3, H, W]
            remain_images_cond = original_rgbs[key_frame][remain_idxs].to(device) # [remain_length, 3, H, W]

            torchvision.utils.save_image(sample_images, f'{tag}_sample_images.png', nrow=sample_images.shape[0], normalize=True)
            torchvision.utils.save_image(sample_images_cond, f'{tag}_sample_images_cond.png', nrow=sample_images_cond.shape[0], normalize=True)

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
            ) # (1, C, f, H, W)

            sample_images_edit = rearrange(sample_images_edit, '1 C f H W -> f C H W').to(device, dtype=torch.float32) # (f, C, H, W)
            if sample_images_edit.shape[-2:] != (H, W):
                sample_images_edit = F.interpolate(sample_images_edit, size=(H, W), mode='bilinear', align_corners=False)

            # spatial warp (based on depth map, see ViCA-NeRF)
            for idx_cur, i in enumerate(sample_idxs):
                warp_average = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
                weights_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
                intrinsic_cur = time_cam_dataset.intrinsics[i]
                extrinsic_cur = time_cam_dataset.extrinsics[i]
                for idx_ref, j in enumerate(sample_idxs):
                    intrinsic_ref = time_cam_dataset.intrinsics[j]
                    extrinsic_ref = time_cam_dataset.extrinsics[j]
                    # The process of warping is :
                    # 1. warp A's pixels to B (find the correspondence)
                    # 2. use B' pixels to paint A
                    # every coordinate of A has its corresponding coordinate in B, which size is 2
                    warp_cur_from_ref = warp_pts_BfromA(intrinsic_cur, extrinsic_cur, key_frame_depth_maps[i], intrinsic_ref, extrinsic_ref) # (H, W, 2)
                    image_ref = sample_images_edit[idx_ref].permute(1, 2, 0).float()
                    image_cur = sample_images_edit[idx_cur].permute(1, 2, 0).float()
                    warp, mask = apply_warp(warp_cur_from_ref, image_cur, image_ref)
                    weight = (mask != 0).sum() / (mask).numel()
                    warp_average[mask] += warp[mask] * weight
                    weights_mask[mask] += weight

                average_mask = (weights_mask != 0)
                warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1)
                sample_images_edit[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask]

            torchvision.utils.save_image(sample_images_edit, f'{tag}_sample_images_edit.png', nrow=sample_images_edit.shape[0], normalize=True)
            torchvision.utils.save_image(remain_images, f'{tag}_remain_images.png', nrow=remain_images.shape[0], normalize=True)




    key_frame_edit(key_frame=0, warp_ratio=0.5, warm_up_steps=2)
    print("Key frame editing done!")

    keyframe_images = time_cam_dataset.time_cam_image[0] # [num_cam, H, W, 3]
    keyframe_images = rearrange(keyframe_images, 'f H W C -> f C H W')  # [num_cam, 3, H, W]
    torchvision.utils.save_image(keyframe_images, f'{tag}_keyframe_images.png', nrow=args.sequence_length, normalize=True)
    

if __name__ == '__main__':
    # Set up argument parser and merge config file(yaml)
    parser = ArgumentParser(description='Render script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="configs/dynerf/coffee_martini.yaml")
    parser.add_argument("--chkpnt_path", type=str, default="output/N3V/coffee_martini/chkpnt_best.pth")
    parser.add_argument("--seed", type=int, default=6666)

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
    
    args = parser.parse_args(args=[])

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