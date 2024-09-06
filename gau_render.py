import os
from os import makedirs
import torch
from torch import nn
import torchvision
from scene import Scene, GaussianModel
from argparse import ArgumentParser
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

if __name__ == '__main__':
    # Set up argument parser and merge config file(yaml)
    parser = ArgumentParser(description='Render script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, default="configs/dynerf/coffee_martini_render.yaml")
    parser.add_argument("--chkpnt_path", type=str, default="output/N3V/coffee_martini/chkpnt_best.pth")
    parser.add_argument("--seed", type=int, default=6666)
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

    gaussians = GaussianModel(args.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if args.eval_shfs_4d else 0)
    scene = Scene(dataset, gaussians, shuffle=False, num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
    (model_param, _) = torch.load(args.chkpnt_path)
    gaussians.restore(model_param, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if pipe.env_map_res:
        env_map = nn.Parameter(torch.zeros((3, pipe.env_map_res, pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(False))
    else:
        env_map = None

    gaussians.env_map = env_map

    video_dataset = scene.getTestCameras()

    render_path = os.path.join(args.model_path, "video", "renders")
    print("Render path: " + render_path)
    gts_path = os.path.join(args.model_path, "video", "gt")
    print("Ground Truth path: " + gts_path)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians.get_xyz.shape[0])
    with torch.no_grad():
        for idx, view in enumerate(tqdm(video_dataset, desc="Rendering progress")):
            if idx == 0:time1 = time()

            gt_image, viewpoint_cam = view
            gt_image = gt_image.cuda()
            viewpoint_cam = viewpoint_cam.cuda()
            gt_list.append(gt_image)

            rendering = render(viewpoint_cam, gaussians, pipe, background)["render"]
            render_images.append(to8b(rendering).transpose(1,2,0))
            render_list.append(rendering)

        time2 = time()
        print("FPS:",(len(video_dataset)-1)/(time2-time1))

        multithread_write(gt_list, gts_path)
        multithread_write(render_list, render_path)

        imageio.mimwrite(os.path.join(args.model_path, "video", 'video_rgb.mp4'), render_images, fps=30)