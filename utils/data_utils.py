import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np
from utils.graphics_utils import getWorld2View2
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from torchvision import transforms as T
from line_profiler import LineProfiler
import ipdb

class CameraDataset(Dataset):
    
    def __init__(self, cam_time_image, viewpoint_stack, white_background, num_frame=300):
        self.num_cam = len(viewpoint_stack) // num_frame
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        self.intrinsics = torch.zeros((self.num_cam, 3, 3), dtype=torch.float32)
        self.extrinsics = torch.zeros((self.num_cam, 4, 4), dtype=torch.float32)

        self.define_transforms()

        tbar = tqdm(range(len(viewpoint_stack)),desc="Reading Images")

        def process_viewpoint(viewpoint_cam, bg):
            if viewpoint_cam.meta_only:
                with Image.open(viewpoint_cam.image_path) as image_load:
                    im_data = np.array(image_load.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image_load = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
                resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
                viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
                if resized_image_rgb.shape[0] == 4:
                    gt_alpha_mask = resized_image_rgb[3:4, ...]
                    viewpoint_image *= gt_alpha_mask
                else:
                    viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
            else:
                viewpoint_image = viewpoint_cam.image
            tbar.update(1)
            return viewpoint_image

        with ThreadPool() as pool:
            results = pool.map(
                lambda viewpoint_cam: process_viewpoint(viewpoint_cam, bg=self.bg),
                viewpoint_stack
            )

        self.time_cam_image = torch.stack(results, 0) # [num_frame * num_cam, C, H, W]

        for i in range(0, len(viewpoint_stack), num_frame):
            self.getIn_Ex_trinsics(i // 300, viewpoint_stack[i])

    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        # if viewpoint_cam.meta_only:
        #     with Image.open(viewpoint_cam.image_path) as image_load:
        #         im_data = np.array(image_load.convert("RGBA"))
        #     norm_data = im_data / 255.0
        #     arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
        #     image_load = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
        #     resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
        #     viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
        #     if resized_image_rgb.shape[0] == 4:
        #         gt_alpha_mask = resized_image_rgb[3:4, ...]
        #         viewpoint_image *= gt_alpha_mask
        #     else:
        #         viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
        # else:
        #     viewpoint_image = viewpoint_cam.image
        viewpoint_image = self.time_cam_image[index]
        return viewpoint_image, viewpoint_cam

    def __len__(self):
        return len(self.viewpoint_stack)
    
    def getIn_Ex_trinsics(self, idx, cam):
        self.intrinsics[idx, 0, 0] = cam.fl_x
        self.intrinsics[idx, 1, 1] = cam.fl_y
        self.intrinsics[idx, 0, 2] = cam.cx
        self.intrinsics[idx, 1, 2] = cam.cy
        self.intrinsics[idx, 2, 2] = 1.0

        self.extrinsics[idx] = torch.from_numpy(getWorld2View2(cam.R, cam.T))

    def define_transforms(self):
        self.transform = T.ToTensor()