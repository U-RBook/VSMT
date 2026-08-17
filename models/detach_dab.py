import random
import torch
from torch import nn
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import linalg


class HEDTransform(nn.Module):
    def __init__(self, dynamic_weight=False):
        super(HEDTransform, self).__init__()

        rgb_from_hed = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
        hed_from_rgb = linalg.inv(rgb_from_hed)
        self.rgb_from_hed = nn.Parameter(torch.tensor(rgb_from_hed, dtype=torch.float32),
                                                       requires_grad=dynamic_weight)                                       
        self.hed_from_rgb = nn.Parameter(torch.tensor(hed_from_rgb, dtype=torch.float32), 
                                         requires_grad=dynamic_weight)


    def convert(self, img, mode):
        if mode == 'rgb2hed':
            return self.rgb2hed(img)
        elif mode == 'hed2rgb':
            return self.hed2rgb(img)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def forward(self, image):

        hed_image = self.convert(image, mode='rgb2hed')  # (1, 3, H, W)

        null = torch.zeros_like(hed_image[:, 0, :, :]) - 1
        ihc_h = self.convert((torch.stack((hed_image[:, 0, :, :], null, null), axis=1)), mode='hed2rgb')
        ihc_d = self.convert((torch.stack((null, null, hed_image[:, 2, :, :]), axis=1)), mode='hed2rgb')
        
        return ihc_h, ihc_d #he and dab

    
    def rgb2hed(self, rgb):

        rgb = (rgb + 1) / 2 # Normalize to [0, 1]
        rgb = torch.clamp(rgb, min=1e-6)
        log_adjust = torch.log(torch.tensor(1e-6, dtype=rgb.dtype, device=rgb.device))
        rgb = torch.log(rgb) / log_adjust
        rgb = rgb.permute(0, 2, 3, 1)
        hed = torch.matmul(rgb, self.hed_from_rgb)
        hed = torch.clamp(hed, min=0)
        hed = hed.permute(0, 3, 1, 2)

        # return hed*2 - 1 # Normalize to [-1, 1]
        return hed
    
    def hed2rgb(self, hed):
        """Convert HED image (B, C, H, W) to RGB color space.

        Parameters
        ----------
        hed : torch.Tensor
            The image in HED format with shape (B, C, H, W).

        Returns
        -------
        rgb : torch.Tensor
            The image in RGB format with shape (B, 3, H, W).
        """
        # hed = (hed + 1) / 2 # Normalize to [0, 1]

        log_adjust = - torch.log(torch.tensor(1e-6, dtype=hed.dtype, device=hed.device))
        hed = hed * log_adjust

        hed = hed.permute(0, 2, 3, 1)
        rgb = - torch.matmul(hed, self.rgb_from_hed)
        rgb = torch.exp(rgb)

        rgb = torch.clamp(rgb, min=0, max=1)
        rgb = rgb.permute(0, 3, 1, 2)

        return rgb*2 - 1 # Normalize to [-1, 1]
        # return rgb
    
    def jitter_channels(self, rgb_img, jitter_d=[0.5,1.5], jitter_h=[0.5,1.5], p=0.5):

        if random.random() < p:
            hed = self.rgb2hed(rgb_img)
            d_factor = random.uniform(jitter_d[0], jitter_d[1])
            h_factor = random.uniform(jitter_h[0], jitter_h[1])
            hed[:, 0, :, :] *= h_factor
            hed[:, 2, :, :] *= d_factor
            rgb_img = self.hed2rgb(hed)
            
        return rgb_img
        
      
        

