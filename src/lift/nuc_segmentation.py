from pathlib import Path
import skimage
import numpy as np
from lift.general_utils import wavelet_filter, load_sequence

#####
#
# below are the helper function to apply the segmentation function to a set of paths or a list of images
#
#####

def segment_imglist(img_list, method, min_area=1000):
    """
    apply nucleus segmentation to a list of images

    parameters:
    img list: the list of images that need to be segmented
    method: class with a class.segment method that perform the segmentation
    min_area: nuclei with an area (in pixels) smaller than this are removed
    """
    
    # segment the nuclei    
    masks = method.segment(img_list)
      
    # post process the segmented nuclei    
    result = []
    for seg_result in masks:
        cleared = skimage.segmentation.clear_border(seg_result)  # remove objects touching the border
    
        for label in range(1, np.max(cleared)):  # filter out small objects
            if np.sum(cleared == label) < min_area:
                cleared[cleared == label] = 0   

        filtered, _, _ = skimage.segmentation.relabel_sequential(cleared)
        result.append(filtered)

    return result


def segment_folderlist(folders_list, method, preproc_function=None, min_area=1000, **kwargs):
    """
    apply nucleus segmentation to a set of folders. The folder should be in a list and point to 
    the experiment you whould like to process. The folder should contain the max projected .tif files 
    from the imaging experiment. Each time point is saved as a separate file

    parameters:
    folder list: the list of folders that need to be segmented
    method: class with a class.segment method that perform the segmentation
    preproc_function: the function used for preprocessing the images before running the segmentation
                        set to None to apply no preprocessing
    min_area: nuclei with an area (in pixels) smaller than this are removed
    **kwargs: the additional arguments for the preprocessing function
    """    
    for folder in folders_list:
        folder = Path(folder) 
        print("processing:", folder)

        # load the data
        raw_paths = list(folder.glob("*.tif"))
        raw_data = load_sequence.load(folder)

        # run the preprocessing step
        processed = preprocessing(raw_data, preproc_function, **kwargs)

        # run the segmentation
        segmented_result = segment_imglist(processed, method, min_area)

        # save the output
        save_folder = folder.parent / "results" / "result_cell_seg"
        save_folder.mkdir(parents=True, exist_ok=True)

        for raw_path, result in zip(raw_paths, segmented_result):
            save_name = save_folder / raw_path.name
            skimage.io.imsave(save_name, result, check_contrast=False)

    
#####
#
# below are the different nucleus segmentation methods
#
#####


class CP_SAM(object):
    """
    cellpose-SAM based nucleus segmentation

    parameters:
    flow-threshold threshold for the error in the flow field, increase to detect more nuclei
    cellprob_threshold probability of detecting a nucleus, decrease to detect more nuclei
    tile_norm_blocksize tile size used for normalizing the images, use 0 for the entire image
    batch_size number of images to process at once. Decrease if less memory is available
    scale_factor: factor for resizing the images. Cellpose-SAM was trained on a nucleus size of 
        7.5 - 120 pixels so rescale to make sure the images fall inside of this range
    """

    def __init__(self, flow_threshold=0.0, cellprob_threshold=-0.5,
                tile_norm_blocksize=0, batch_size=8, scale_factor=1):

        from lift._helpers import _require_cellpose_sam
        _require_cellpose_sam()
        from cellpose import models
    
        self.fT = flow_threshold
        self.cT = cellprob_threshold
        self.tnb = tile_norm_blocksize
        self.bs = batch_size
        self.scale = scale_factor

        self.cpsam = models.CellposeModel(gpu=True)


    def segment(self, img_list):

        # down scale the images using the specified factor
        if self.scale != 1:
            img_list = [skimage.transform.rescale(img, self.scale, order=1) for img in img_list]
        
        # segment the nuclei    
        masks, flows, styles = self.cpsam.eval(img_list, batch_size=self.bs, flow_threshold=self.fT, 
                                          cellprob_threshold=self.cT, normalize={"tile_norm_blocksize": self.tnb})

        if self.scale != 1:
            masks = [skimage.transform.rescale(img, 1/self.scale, order=0) for img in masks]

        return masks


class CP_V3(object):
    """
    cellpose v3 based nucleus segmentation mostly usefull for when no GPU is available to run cellpose-SAM

    parameters:
    flow-threshold threshold for the error in the flow field, increase to detect more nuclei
    cellprob_threshold probability of detecting a nucleus, decrease to detect more nuclei
    diameter size of the nuclei in pixels that you wish to detect

    """

    def __init__(self, flow_threshold=0.0, cellprob_threshold=-0.5, diameter=140):
        from lift._helpers import _require_cellpose_v3
        _require_cellpose_v3()
        from cellpose import models
        import torch

        self.fT = flow_threshold
        self.cT = cellprob_threshold
        self.diam = diameter

        gpu_available = True if torch.cuda.is_available() else False
        self.cp = models.CellposeModel(gpu=gpu_available, model_type="nuclei")

    
    def segment(self, img_list):

        # segment the nuclei    
        masks, flows, styles = self.cp.eval(img_list, flow_threshold=self.fT, cellprob_threshold=self.cT,
                                          diameter=self.diam, channels="Grayscale", normalize=True)

        return masks
        

#####
#
# below are the different preprocessing function to apply
#
#####

    
def preprocessing(img_list, preproc_method, **kwargs):
    # preprocess the images to enhance performance and make the images look more like a DAPI staining
    
    processed = []
    for I in img_list:   
        if preproc_method:
            proc = preproc_method(I, **kwargs)
            processed.append(proc)
        else:
            processed.append(I.copy())

    return processed


def contrast_adjuster(I, **kwargs):
    """
    apply blurring remove high frequencies (the foci) from the images and clip the images to make the
    background signal in the nucleus brighter
    """
    sigma = kwargs.pop("sigma", 6)
    factor = kwargs.pop("c_factor", 0.4)
    
    blurred = skimage.filters.gaussian(I, sigma=sigma)
    #scaled = np.clip(blurred / (np.max(blurred)*factor), 0, 1)         
    scaled = np.clip(blurred / ((np.mean(blurred)*factor)), 0, 1)
    
    return scaled


def wavelet_filtering(I, **kwargs):
    """
    use wavelet filtering to remove high frequencies (the foci) from the images
    """
    scales = kwargs.pop("scales", 10)
    factor = kwargs.pop("w_factor", 2.2)
    start_scale = kwargs.pop("start_scale", 2)

    W, I = wavelet_filter.wavelets(I, scales=scales)       
    for w in W:
      threshold = factor * (np.std(w))**2
      w[w**2 > threshold] = 0
    coefs = np.sum(np.stack(W[start_scale:]),0)
    recon = I[-1] + coefs

    recon = skimage.util.img_as_ubyte(recon/recon.max())

    return recon
    