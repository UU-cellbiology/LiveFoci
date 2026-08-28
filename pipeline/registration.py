import os
from pathlib import Path
import subprocess
import numpy as np
import skimage
from pystackreg import StackReg
from pystackreg.util import to_uint16
from pipeline.general_utils.wavelet_filter import wavelets

#####
#
# below are the helper functions for running all different registration options and saving the output
#
#####


def register_array(stack, method, preprocess_function=None, **kwargs):

    """
    performs registration on a single stack using the specified registration method 
    with the corresponding preprocessing. 
    
    stack: numpy array for the image of shape t, width, height

    method: string specifying which registration method to use. The function should take a numpy array
    as input images in which the first dimension corresponds to the time axis.
    
    preprocess_function: python function that performs preprocessing of the numpy array and return
    a new numpy array. Set to None to use the raw input stack directly

    """

    if method=='stackreg':
        registered_stack = stackreg_registration(stack, preprocess_function)
    elif method=='elastix':
        loss = kwargs.pop("loss", "MSE")
        elastix_object = ElastixReg(loss, preprocess_function)
        registered_stack = elastix_object.register_stack(stack)
    else:
        raise ValueError(f"Unknown registration method: {method!r}")

    return registered_stack


def run_registration(path_list, method, preprocess_function=None, save_name='registered', **kwargs):
    """
    performs registration of a stack of images using the specified registration method 
    with the corresponding preprocessing. 
    
    path_list: should be a list of paths of the raw stacks that should be registered. 

    method: string specifying which registration method to use. The function should take a numpy array
    as input images in which the first dimension corresponds to the time axis.
    
    preprocess_function: python function that performs preprocessing of the numpy array and return
    a new numpy array. Set to None to use the raw input stack directly

    save_name: optional name for saving the registered output    
    """
    if preprocess_function:
        print(f"running {method} registration with {preprocess_function.__name__} as preprocessing function")
    else:
        print(f"running {method} registration without preprocessing")

    for path in path_list:
        stack = skimage.io.imread(path)
    
        registered_stack = register_array(stack, method, preprocess_function, **kwargs)  

        # get the correct names for saving
        path = Path(path)
        tifname = path.name  # filename
        nucleus_id = path.parent.name  # folder containing the file
        pos_dir = path.parents[2]  # grandparent of parent (3 levels up)
        save_dir = pos_dir / save_name / nucleus_id

        # make the directory for saving
        save_dir.mkdir(parents=True, exist_ok=True) 

        # save the registered stack
        save_path = save_dir / tifname
        skimage.io.imsave(save_path, registered_stack, check_contrast=False)

    
#####
#
# below are the different registration methods that are implemented
#
#####

    
def stackreg_registration(stack, preproc_func=None):   
    sr = StackReg(StackReg.RIGID_BODY)

    if preproc_func:
        preprocessed_stack = preproc_func(stack)
        tmats = sr.register_stack(preprocessed_stack, reference='previous')
        registered_stack = sr.transform_stack(stack, tmats=tmats)
    else:  
        registered_stack = sr.register_transform_stack(stack, reference='previous')
    
    info = np.iinfo(stack.dtype)
    registered_stack = np.clip(to_uint16(registered_stack), info.min, info.max).astype(stack.dtype)

    return registered_stack


class ElastixReg(object):
    
    def __init__(self, loss='MSE', preprocess_function=None, previous_initialisation=True):
        super().__init__()
        self.preproc_function = preprocess_function
        self.prev_init = previous_initialisation
        
        #get directory for this class
        self.wd = Path(__file__).resolve().parent   # Path.cwd() 
        elastix_dir = self.wd / "utils_elastix"
        
        self.elastix_bin = elastix_dir  / "elastix.exe"
        self.param_loc = elastix_dir  / f"elastix_parameters_{loss}.txt"
        self.out_dir = elastix_dir / "Temp"

        self.fixed = elastix_dir  / "Temp" / "fixed.tif"
        self.moving = elastix_dir  / "Temp" / "moving.tif"  

        self.transform_loc = self.out_dir / "prev_transforms"  # location for transform of previous frames for initialisation
        
        if self.preproc_function:  # change the command to support multi image registration
            self.fixed_processed = elastix_dir  / "Temp" / "fixed_processed.tif"
            self.moving_processed = elastix_dir  / "Temp" / "moving_processed.tif"
            self.param_loc = elastix_dir  / f"elastix_parameters_MultiImage_{loss}.txt"
            self.cmd = [
                str(self.elastix_bin),
                "-f0", str(self.fixed),
                "-m0", str(self.moving),
                "-f1", str(self.fixed_processed),
                "-m1", str(self.moving_processed),
                "-p", str(self.param_loc),
                "-out", str(self.out_dir),
            ]
        else:    
            self.cmd = [
                str(self.elastix_bin),
                "-f", str(self.fixed),
                "-m", str(self.moving),
                "-p", str(self.param_loc),
                "-out", str(self.out_dir),
            ]

        self._validate_paths()

    
    def _validate_paths(self):
        if not self.elastix_bin.exists():
            raise FileNotFoundError(f"Elastix binary not found: {self.elastix_bin}")

        if not self.param_loc.exists():
            raise FileNotFoundError(f"Parameter file not found: {self.param_loc}")
            
    
    def register_stack(self, img_stack):

        elastix_reg = [img_stack[0].copy()]
        skimage.io.imsave(self.fixed, elastix_reg[-1], check_contrast=False)

        if self.preproc_function:
            processed_stack = self.preproc_function(img_stack)
            skimage.io.imsave(self.fixed_processed, processed_stack[0], check_contrast=False)
            skimage.io.imsave(self.moving_processed, processed_stack[1], check_contrast=False)  

        timepoints, nr, nc = img_stack.shape
        for tt in range(timepoints-1):   
            skimage.io.imsave(self.moving, img_stack[tt+1], check_contrast=False)

            if tt > 0 and self.prev_init:
                prev_transform = self.transform_loc / f"TransformParameters.{tt-1}.txt"
                (self.out_dir / "TransformParameters.0.txt").replace(prev_transform)
                reg_command = self.cmd + ["-t0", str(prev_transform)]
            else:
                reg_command = self.cmd
            
            p = subprocess.run(reg_command, capture_output=False, text=True)
            #print(p.stderr)
            #print(p.stdout)

            registered_image = skimage.io.imread(self.out_dir / "result.0.tif")
            elastix_reg.append(registered_image)

            # update the fixed image using the registered result
            (self.out_dir / "result.0.tif").replace(self.fixed)

            # process the registered image again and update the temporary processed images
            if self.preproc_function:
                _info = np.iinfo(img_stack.dtype)
                processed_reg = wavelet_denoise([np.clip(registered_image, _info.min, _info.max).astype(img_stack.dtype)])
                skimage.io.imsave(self.fixed_processed, processed_reg, check_contrast=False)
                skimage.io.imsave(self.moving_processed, processed_stack[tt+1], check_contrast=False)  
        
        elastix_reg = np.stack(elastix_reg)
        _info = np.iinfo(img_stack.dtype)
        elastix_reg = np.clip(elastix_reg, _info.min, _info.max).astype(img_stack.dtype)

        # remove the files of the previous transforms
        for f in self.transform_loc.glob("TransformParameters.*.txt"):
            f.unlink()

        return elastix_reg



#####
#
# below are different pre processing functions for the registration
#
#####

def wavelet_denoise(stack):
    num_scales = 2
    factor = 5.0
    start_scale = 1
    
    preprocessed_stack = []
    for frame in stack:
      W, I = wavelets(frame, scales=num_scales)

      for w in W:
        threshold = factor*(np.std(w))**2
        w[w**2 < threshold] = 0
    
      coefs = np.sum(np.stack(W[start_scale:]),0)
      recon_full = 0*I[-1] + coefs
      recon_full = np.clip(recon_full, 0, None)
    
      preprocessed_stack.append(recon_full)
    preprocessed_stack = np.stack(preprocessed_stack)

    return preprocessed_stack


def DOG_filter(stack):
    sig_low = 0.75
    sig_high = 1.0
    
    preprocessed_stack = []
    for frame in stack:
        processed = skimage.filters.difference_of_gaussians(frame.astype(np.float32), low_sigma=sig_low, high_sigma=sig_high)
        
        preprocessed_stack.append(processed)
    preprocessed_stack = np.stack(preprocessed_stack)

    return preprocessed_stack

        
def threshold(stack):
    rel_threshold = 0.1
    
    preprocessed_stack = []
    for frame in stack:
        processed = frame.copy()
        processed[frame < processed.max()*rel_threshold] = 0
        
        preprocessed_stack.append(processed)
    preprocessed_stack = np.stack(preprocessed_stack)

    return preprocessed_stack