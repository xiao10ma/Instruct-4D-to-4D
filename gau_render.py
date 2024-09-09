# import os
# from os import makedirs
# import torch
# from torch import nn
# import torchvision
# from scene import Scene, GaussianModel
# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, OptimizationParams
# from omegaconf import OmegaConf
# from omegaconf.dictconfig import DictConfig
# import numpy as np
# import random
# from tqdm import tqdm
# from time import time
# from gaussian_renderer import render
# import concurrent.futures
# import cv2
# import colorsys
# from sklearn.decomposition import PCA
# from PIL import Image

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



# def feature_to_rgb(features):
#     # Input features shape: (16, H, W)
    
#     # Reshape features for PCA
#     H, W = features.shape[1], features.shape[2]
#     features_reshaped = features.view(features.shape[0], -1).T

#     # Apply PCA and get the first 3 components
#     pca = PCA(n_components=3)
#     pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

#     # Reshape back to (H, W, 3)
#     pca_result = pca_result.reshape(H, W, 3)

#     # Normalize to [0, 255]
#     pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

#     rgb_array = pca_normalized.astype('uint8')

#     return rgb_array

# def id2rgb(id, max_num_obj=256):
#     if not 0 <= id <= max_num_obj:
#         raise ValueError("ID should be in range(0, max_num_obj)")

#     # Convert the ID into a hue value
#     golden_ratio = 1.6180339887
#     h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
#     s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
#     l = 0.5

    
#     # Use colorsys to convert HSL to RGB
#     rgb = np.zeros((3, ), dtype=np.uint8)
#     if id==0:   #invalid region
#         return rgb
#     r, g, b = colorsys.hls_to_rgb(h, l, s)
#     rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

#     return rgb

# def visualize_obj(objects):
#     rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
#     all_obj_ids = np.unique(objects)
#     for id in all_obj_ids:
#         colored_mask = id2rgb(id)
#         rgb_mask[objects == id] = colored_mask
#     return rgb_mask


# if __name__ == '__main__':
#     # Set up argument parser and merge config file(yaml)
#     parser = ArgumentParser(description='Render script parameters')
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#     parser.add_argument("--config", type=str, default="configs/dynerf/coffee_martini_render.yaml")
#     parser.add_argument("--chkpnt_path", type=str, default="output/N3V/coffee_martini/chkpnt_best.pth")
#     parser.add_argument("--seed", type=int, default=6666)
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

#     gaussians = GaussianModel(args.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if args.eval_shfs_4d else 0)
#     scene = Scene(dataset, gaussians, shuffle=False, num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)
#     (model_param, _) = torch.load(args.chkpnt_path)
#     gaussians.restore(model_param, opt)
    
#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
#     if pipe.env_map_res:
#         env_map = nn.Parameter(torch.zeros((3, pipe.env_map_res, pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(False))
#     else:
#         env_map = None

#     gaussians.env_map = env_map

#     video_dataset = scene.getTestCameras()

#     num_classes = dataset.num_classes

#     classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
#     classifier.cuda()
#     classifier.load_state_dict(torch.load(os.path.join(scene.model_path, f"chkpnt_best_classifier.pth")))
    


#     render_path = os.path.join(args.model_path, "video", "render_rgb") # RGB
#     print("Render path: " + render_path)
#     gts_path = os.path.join(args.model_path, "video", "gt_rgb") # RGB
#     print("Ground Truth path: " + gts_path)
#     colormask_path = os.path.join(args.model_path, "video", "objects_feature16") # feature 16
#     print("Gaussian Object Spalatting path: ", colormask_path)
#     gt_colormask_path = os.path.join(args.model_path, "video", "gt_objects_color") # rgb, 但是是 SAM 得到的 mask
#     print("Ground Truth Mask path: " + gt_colormask_path)
#     pred_obj_path = os.path.join(args.model_path, "video", "objects_pred") # rgb, feature 16 -> 256, argmax, 得到的 mask
#     print("Gaussian Spalatting Classified path: ", pred_obj_path)
    
#     makedirs(gts_path, exist_ok=True)
#     makedirs(render_path, exist_ok=True)
#     makedirs(colormask_path, exist_ok=True)
#     makedirs(gt_colormask_path, exist_ok=True)
#     makedirs(pred_obj_path, exist_ok=True)
    
#     to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
#     render_images = []
#     gt_rgb_list = []
#     render_rgb_list = []
#     gt_mask_list = []
#     pred_mask_list = []
#     splat_list = []
#     print("point nums:",gaussians.get_xyz.shape[0])
#     with torch.no_grad():
#         for idx, view in enumerate(tqdm(video_dataset, desc="Rendering progress")):
#             if idx == 0:time1 = time()

#             gt_image, viewpoint_cam = view
#             gt_image = gt_image.cuda()
#             viewpoint_cam = viewpoint_cam.cuda()

#             results = render(viewpoint_cam, gaussians, pipe, background)
#             rendering, rendering_obj = results["render"], results["render_object"]

#             logits = classifier(rendering_obj)
#             pred_obj = torch.argmax(logits, dim=0)
#             pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8)) # mask (gaussian spalatting 16 -> 256, argmax, 得到分类)

#             gt_objects = viewpoint_cam.objects
#             gt_objects = gt_objects.squeeze(0)
#             gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8)) # mask (SAM)

#             rgb_mask = feature_to_rgb(rendering_obj) # splatting -> rgb

#             gt_rgb_list.append(gt_image)
#             render_rgb_list.append(rendering)
#             Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
#             Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
#             Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))


#         time2 = time()
#         print("FPS:",(len(video_dataset)-1)/(time2-time1))

#         multithread_write(gt_rgb_list, gts_path)
#         multithread_write(render_rgb_list, render_path)


#         out_path = os.path.join(render_path[:-10],'concat')
#         makedirs(out_path,exist_ok=True)
#         fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
#         size = (gt_image.shape[-1]*5, gt_image.shape[-2])
#         fps = float(5) if 'train' in out_path else float(1)
#         writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

#         for file_name in sorted(os.listdir(gts_path)):
#             gt = np.array(Image.open(os.path.join(gts_path,file_name)))
#             rgb = np.array(Image.open(os.path.join(render_path,file_name)))
#             gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
#             render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
#             pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

#             result = np.hstack([gt,rgb,gt_obj,pred_obj,render_obj])
#             result = result.astype('uint8')

#             Image.fromarray(result).save(os.path.join(out_path,file_name))
#             writer.write(result[:,:,::-1])

#         writer.release()

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
    parser.add_argument("--config", type=str, default="configs/dynerf/coffee_martini.yaml")
    parser.add_argument("--chkpnt_path", type=str, default="output/N3V/coffee_martini/edit_chkpnt_200.pth")
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