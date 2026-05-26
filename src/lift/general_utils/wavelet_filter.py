# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:28:45 2025

@author: 084011
"""

from glob import glob
import numpy as np

def get_kernel(scale, base_kernel=None):
    import torch
    import torch.nn.functional as F
    if base_kernel is None:
        base_kernel = torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
    non_zero_idx = np.arange(0, torch.numel(base_kernel), 1) * scale
    kernel_1D = torch.zeros(non_zero_idx[-1] + 1)
    kernel_1D[non_zero_idx] = base_kernel
    kernel_1D = kernel_1D.unsqueeze(0)
    kernel_2D = (kernel_1D * kernel_1D.T).view(1, 1, torch.numel(kernel_1D), torch.numel(kernel_1D))
    return kernel_2D

def get_scale_k(image, k):
    import torch
    import torch.nn.functional as F

    kernel = get_kernel(k)
    ps = kernel.shape[-1] // 2
    padded = F.pad(image, (ps, ps, ps, ps), mode='reflect')
    i_k = F.conv2d(padded, kernel, padding=0)
    wave_k = image - i_k

    return wave_k, i_k


def wavelets(img, scales=3):
    import torch
    img = torch.tensor(img).unsqueeze(0).to("cpu").float()

    W, I = [], []
    for k in range(1, scales+1):
        wk, img = get_scale_k(img, k)
        W.append(wk.detach().cpu().squeeze().numpy())
        I.append(img.detach().cpu().squeeze().numpy())

    return W, I
