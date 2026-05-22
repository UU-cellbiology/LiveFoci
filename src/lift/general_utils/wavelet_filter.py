# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 16:28:45 2025

@author: 084011
"""

from glob import glob
import numpy as np
import skimage
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def get_kernel(scale, base_kernel=torch.tensor([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])):
    device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    non_zero_idx = np.arange(0, torch.numel(base_kernel), 1) * scale
    kernel_1D = torch.zeros(non_zero_idx[-1] + 1)
    kernel_1D[non_zero_idx] = base_kernel
    kernel_1D = kernel_1D.unsqueeze(0)  # add extra dimension to transpose and multiply in next line
    kernel_2D = (kernel_1D * kernel_1D.T).view(1, 1, torch.numel(kernel_1D), torch.numel(kernel_1D))

    return kernel_2D.to(device)


def get_scale_k(image, k):

    kernel = get_kernel(k)
    ps = kernel.shape[-1] // 2
    padded = F.pad(image, (ps, ps, ps, ps), mode='reflect')
    i_k = F.conv2d(padded, kernel, padding=0)
    wave_k = image - i_k

    return wave_k, i_k


def wavelets(img, scales=3):
    img = torch.tensor(img).unsqueeze(0).to("cpu").float()

    W, I = [], []
    for k in range(1, scales+1):
        wk, img = get_scale_k(img, k)
        W.append(wk.detach().cpu().squeeze().numpy())
        I.append(img.detach().cpu().squeeze().numpy())

    return W, I

if __name__ == "__main__":
    file = r"C:\Users\084011\Documents\projects\live_cell_foci_analysis\Ho_Y_data\test_method\data\20240717_Y_15_Gy\Position004\0200.tif"
    
    img = skimage.io.imread(file) 
    plt.imshow(img, cmap='gray', vmax=0.6*img.max())
    
    W, I = wavelets(img, scales=5)

    start_scale = 0
    factor = 2.0
    for w in W:
      threshold = factor * (np.std(w))**2
      w[w**2 > threshold] = 0
    coefs = np.sum(np.stack(W[start_scale:]),0)
    recon_full = I[-1] + coefs
    
    fig, ax = plt.subplots(nrows=1, ncols=3, dpi=500, layout='tight')
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(I[-1], cmap='gray')
    ax[2].imshow(recon_full, cmap='gray')
    
    for axis in ax:
      axis.axis('off')