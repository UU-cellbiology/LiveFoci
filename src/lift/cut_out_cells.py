#cut_out_cells.py

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:04:22 2024

@author: 084011
"""

import numpy as np
from pathlib import Path
import skimage
from tifffile import imwrite
import os
from glob import glob
from xml.dom import minidom 
import string
import random
import matplotlib.pyplot as plt
from lift.general_utils import load_sequence



def cut_from_xml(path, time_points, box_size):
    """
    this is an old version of the code that directly used the tracking xml
    files to cut out the cells and should no longer be used
    """

    instance_name = os.path.basename(glob(path + r"/raw/*.tif")[0])[:-8]
    tracks = path + r"/cell_tracking/cell_tracking.NND.xml"
    save_path = path + r"/followed"
    
    
    if os.path.exists(save_path):
        raise ValueError(f"the path: {save_path} already exists. Running the code again will keep adding more followed cells resulting in multiple instances of the same cell but with a different random tag for its name")  
    else:
        os.mkdir(save_path) 
    
    """
    # load the tif image and mask sequence in the correct time order
    """
    img_stack = []
    mask_stack = []
    for tt in range(time_points):
        name = instance_name + f"{tt:04}" + ".tif"
        
        img_path = path + r"/raw/" + name
        img = skimage.io.imread(img_path) #intensity image loaded as x, y
        img_stack.append(img)
            
        seg_path = path + r"/result_cell_seg/" + name
        seg = skimage.io.imread(seg_path) #intensity image loaded as x, y
        mask_stack.append(seg)
    
    
    
    tree = minidom.parse(tracks)
    tracks = tree.getElementsByTagName('particle')
    for track in tracks:
        cell = []
        
        detections = track.getElementsByTagName('detection')
        t_start = int(detections[0].attributes['t'].value)
        t_end = int(detections[-1].attributes['t'].value)
        for nn in range(t_end - t_start + 1):
            x = int(float(detections[nn].attributes['x'].value))
            y = int(float(detections[nn].attributes['y'].value))
            t = t_start + nn
            
            #extra check because tracking still allows for gaps
            #but this break means track can be short than set in the parameter file
            t_track = int(detections[nn].attributes['t'].value)
            if t != t_track:
                t_end = t
                break
            
            # Calculate box bounds
            y_min = y - box_size // 2
            y_max = y + box_size // 2 
            x_min = x - box_size // 2
            x_max = x + box_size // 2 
            
            # Create an empty canvas for the cropped region
            I = np.zeros((box_size, box_size), dtype=img_stack[t].dtype)
            M = np.zeros((box_size, box_size))
            
            # Determine the valid region inside the original image
            img_y_min = max(0, y_min)
            img_y_max = min(img_stack[t].shape[0], y_max)
            img_x_min = max(0, x_min)
            img_x_max = min(img_stack[t].shape[1], x_max)
            
            # Determine the corresponding region in the cropped box
            crop_y_min = max(0, -y_min)
            crop_y_max = crop_y_min + (img_y_max - img_y_min)
            crop_x_min = max(0, -x_min)
            crop_x_max = crop_x_min + (img_x_max - img_x_min)
            
            # Copy the valid region from the original image to the cropped box
            I[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = img_stack[t][img_y_min:img_y_max, img_x_min:img_x_max].copy()
            M[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = mask_stack[t][img_y_min:img_y_max, img_x_min:img_x_max].copy()
                    
            # remove signal outside nucleus based on the mask
            M[M != M[box_size//2,box_size//2]] = 0
            I[M == 0] = 0
            
            cell.append(I)
          
            
        cell = np.stack(cell)  
            
        rand_name = "I_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        os.mkdir(f"{save_path}/{rand_name}")
        imwrite(f"{save_path}/{rand_name}/{rand_name}_{t_start:04}_{t_end:04}.tif", cell, imagej=False)
        #skimage.io.imwrite(f"{save_path}/{rand_name}/{rand_name}_{t_start:04}_{t_end:04}.tif", cell)


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


if __name__ == "__main__":  
    #paths = r"..\EBRT_vs_PRRT\20*\Position*"
    paths = r"..\Ho_Y_data\test_method\data\20241120_Xray_25_Gy\ground_truth\cut_out_GT"
    
    paths = glob(paths)        
    for path in paths:
        cut_from_ctc(path)
        #box_size = 140
        #cut_from_xml(path, time_points=360, box_size=box_size)
    