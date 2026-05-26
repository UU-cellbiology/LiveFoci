# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:04:22 2024

@author: 084011
"""

import numpy as np
from pathlib import Path
import skimage
from tifffile import imwrite
import string
import random
import matplotlib.pyplot as plt
from lift.general_utils import load_sequence

def cut_from_ctc(path, margin=30):
    """
    make crops of individual cells over time based on the input defined 
    accordinging to the cell tracking challange format
    """

    path = Path(path)   
    raw_imgs_path = path / "raw" 
    tracks_path = path / "results" / "cell_tracking"
    save_path = path / "results"/ "followed"
    
      
    if save_path.exists():
        raise ValueError(f"The path {save_path} already exists. "
            "Running the code again will add more followed cells resulting in duplicates.")
    else:
        save_path.mkdir(parents=True, exist_ok=True)
    
    
    # load the tif image and mask sequence in the correct time order
    img_stack = load_sequence.load(raw_imgs_path)
    mask_stack = load_sequence.load(tracks_path)
  
    
    # use the tracking information to cut out single cells over time  
    track_file = np.loadtxt(tracks_path / "res_track.txt")
    for track in track_file:
        cell = []
        
        track_label = int(track[0])
        t_start = int(track[1])
        t_end = int(track[2])
        for t in range(t_start, t_end + 1):            
            props = skimage.measure.regionprops(mask_stack[t])
            
            # look for the cell label and cut out the cell
            for prop in props:
                if prop.label == track_label:
                    # bounding box (exclusive max indices)
                    min_row, min_col, max_row, max_col = prop.bbox
            
                    # crop image and mask
                    crop_img = img_stack[t][min_row:max_row, min_col:max_col].copy()
                    crop_mask = (mask_stack[t][min_row:max_row, min_col:max_col] == track_label)
            
                    # zero out everything outside the mask
                    crop_img[crop_mask == 0] = 0
                    cell.append(crop_img)
            
            
            # make sure all crops have the same size
            max_size = np.array([c.shape for c in cell]).max()
            max_size = max_size + 1 if (max_size % 2 != 0) else max_size  # make sure size is even

            box_size = max_size + margin
            padded_cell = []
            for crop_img in cell:
                h, w = crop_img.shape
        
                # padding needed
                pad_y = box_size - h
                pad_x = box_size - w
        
                pad_top = pad_y // 2
                pad_bottom = pad_y - pad_top
                pad_left = pad_x // 2
                pad_right = pad_x - pad_left
        
                # zero-pad to target size
                padded = np.pad(
                    crop_img,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0
                )
        
                padded_cell.append(padded)
            
        cell = np.stack(padded_cell) 

        # Save each followed cell with a random ID
        rand_name = "I_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        rand_dir = save_path / rand_name
        rand_dir.mkdir(parents=True, exist_ok=True)
        imwrite(rand_dir / f"{rand_name}_{t_start:04}_{t_end:04}.tif", cell, imagej=False)