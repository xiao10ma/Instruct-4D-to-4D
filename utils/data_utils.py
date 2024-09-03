import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np
from utils.graphics_utils import getWorld2View2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background, num_frame=300):
        self.num_cam = len(viewpoint_stack) // num_frame
        self.viewpoint_stack = viewpoint_stack
        self.time_cam_image = torch.zeros(num_frame, self.num_cam, 3, viewpoint_stack[0].image_height, viewpoint_stack[0].image_width)
        self.poses = []
        self.bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        self.intrinsics = torch.zeros((self.num_cam, 3, 3), dtype=torch.float32)
        self.extrinsics = torch.zeros((self.num_cam, 4, 4), dtype=torch.float32)
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda args: self.process_viewpoint(*args, num_frame=num_frame, bg=self.bg),
                enumerate(viewpoint_stack)
            )

            for timestamp, cam_idx, viewpoint_image in tqdm(results, total=len(viewpoint_stack), desc="Processing viewpoints"):
                self.time_cam_image[timestamp][cam_idx] = viewpoint_image

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
        viewpoint_image = self.time_cam_image[index%self.num_cam][index/self.num_frame]
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

    def process_viewpoint(self, idx, viewpoint_cam, num_frame, bg):
        if idx % num_frame == 0:
            self.getIn_Ex_trinsics(idx // num_frame, viewpoint_cam)

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

        return int(viewpoint_cam.image_name[-4:]), idx // 300, viewpoint_image