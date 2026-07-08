from pathlib import Path
import numpy as np
import skimage
import scipy
from pipeline.general_utils.wavelet_filter import wavelets


#####
#
# below are the helper functions to run the different foci detection algorithms on numpy arrays or 
# files and save the resulting output
#
#####

def run_detector_stack(stack, detector, threshold, return_segmentation=False):
    """
    Helper function to run the specified detector on a numpy array of the stack of images.
    the shape of the image should be time x width x height

    detector is the corresponding class with the detection module and threshold is the used
    value to look for local maxima

    returns the image with the spot enhanced signals and a list of numpy arrays with detections for each timepoint
    the numpy arrays adheres to the format of the particle tracking challange
    """

    detections = []
    enhanced_stack = []
    label_stack = []
    for tt, frame in enumerate(stack):

        coords, enhanced, labels = detector.detect(frame, threshold, return_segmentation)
        y, x = coords[:,0], coords[:,1]

        if coords.size > 0: 
            intensities = frame[coords[:, 0].astype(int), coords[:, 1].astype(int)]
            coords_with_t = np.column_stack([x, y, 
                                             np.full(len(coords), tt), 
                                             intensities])
        else:
            coords_with_t = np.empty((0, 4))
        detections.append(coords_with_t)

        enhanced_stack.append(enhanced)
        label_stack.append(labels)

    return detections, np.stack(enhanced_stack), np.stack(label_stack)


def run_detection(path_list, detector, threshold, return_segmentation=False):
    """
    Helper function to run the specified detector on a list of paths pointing to images saved as stacks

    detector is the corresponding class with the detection module and threshold is the used
    value to look for local maxima

    the detections are saved as detections.txt next to the image stack
    """
    
    for path in path_list:
        path = Path(path)  # convert to Path object
        stack = skimage.io.imread(path)

        # run the detector on the image stack
        detections, _, label_stack = run_detector_stack(stack, detector, threshold, return_segmentation)

        # convert the list to one large numpy array and save it
        all_detections = np.vstack(detections)

        save_path = path.parent / "detected_foci.txt"
        np.savetxt(save_path, all_detections, fmt='%.2f', delimiter='\t')

        if return_segmentation:
            save_path = path.parent / "spot_segmentation.tif"
            skimage.io.imsave(save_path, label_stack.astype(np.uint16), check_contrast=False)



#####
#
# below is the helper function to load different detectors and the variable containing
# the default parameters for each detector
#
#####


def create_detector(name, params=None, threshold=None):
    """
    Helper function to create the different detectors based on the provided name
    and load the default parameters or use the parameters provided in this function
    """

    DETECTOR_REGISTRY = {
        "Wavelets": {
            "class": WaveletDetector,
            "default_params": {
                "K": 3,
                "factor": 4.0,
                "start_scale": 0
            },
            "default_threshold": 30
        },
        "LOG": {
            "class": LOGDetector,
            "default_params": {
                "sigma": 0.8
            },
            "default_threshold": 26
        },
        "Hessian": {
            "class": HessianDetector,
            "default_params": {
                "sigma": 0.5
            },
            "default_threshold": 5
        },
        "TopHat": {
            "class": TopHatDetector,
            "default_params": {
                "sigma": 0.6,
                "radius": 2.0
            },
            "default_threshold": 26
        },
        "HDome": {
            "class": HDomeDetector,
            "default_params": {
                "h": 50
            },
            "default_threshold": 40
        },
        "HDome-smal": {
            "class": HDomeSmalDetector,
            "default_params": {
                "sigma": 1.2,
                "h": 80,
                "s": 1.5
            },
            "default_threshold": 20
        },
        "MPHD": {
            "class": MPHDDetector,
            "default_params": {
                "sigma": 0.5,
                "h_init": 5,
                "R": 10
            },
            "default_threshold": 20
        },
        "Spotiflow": {
            "class": SpotiflowDetector,
            "default_params": {
                "model_name": "general"
            },
            "default_threshold": 0.55
        }
    }
    

    config = DETECTOR_REGISTRY[name]

    DetectorClass = config["class"]

    default_params = config["default_params"].copy()

    if params:
        default_params.update(params)

    if threshold is None:
        threshold = config["default_threshold"]

    detector = DetectorClass(**default_params)

    return detector, threshold


#####
#
# functions to transform the signal enhanced images into detected coordinates
# both function could be improved by returning the center of mass 
#
#####

def detect_maxima(image, threshold, return_segmentation=False):
    """
    first look for local maxima and possibly also return a segmentation in which the local maxima
    above threshold are used as markers in the watershed to create segmentations of connected components
    """
    
    detections = skimage.feature.peak_local_max(image, threshold_abs=threshold, min_distance=1)
    
    if return_segmentation:
        labels = segment_spots(detections, image, threshold)
    else:
        labels = None
    return detections, labels


def segment_spots(detections, image, threshold):
    """
    the local maxima in 'detections' are used as markers in the watershed to create segmentations of connected components
    the area above 'threshold' in the 'image' is what will be segmented
    """
    detected_pixel = np.round(detections).astype(int) 

    # mask
    mask = image > threshold
    mask[tuple(detected_pixel.T)] = True

    markers = np.zeros(image.shape, dtype=int)
    markers[tuple(detected_pixel.T)] = np.arange(1, len(detected_pixel)+1)

    labels = skimage.segmentation.watershed(-image, markers=markers, mask=mask, watershed_line=True)

    return labels

    
def signal_thresholding(image, threshold):
    """
    segment the image based on threshold and this to create segmentations based on connected components
    """
    
    mask = image > threshold
    labels = skimage.measure.label(mask, background=0, connectivity=1)
    props = skimage.measure.regionprops(labels)
    detections = np.array([prop.centroid for prop in props])
    
    return detections, labels


#####
#
# below are the different foci detectors
#
#####

class BaseDetector:
    """
    base class for the different spot detections methods.
    Depending on the different methods it can overwrite the enhance or detect functions differently
    """

    name = "base"

    def __init__(self, **params):
        self.params = params

    def enhance(self, image):
        raise NotImplementedError

    def detect(self, image, threshold, return_segmentation=False):

        enhanced = self.enhance(image)

        coords, segmentation = detect_maxima(enhanced, threshold, return_segmentation)

        return coords, enhanced, segmentation


class WaveletDetector(BaseDetector):
    """
    Wavelet based enhancing of the spots using Jeffreys non informative prior

    parameters:
    K: integer, number of scales to use in the wavelet decomposition
    factor: used to select significant coefficients as factor * std(wavelet coefficients)
    start_scale: integer, the scale to use as first scale in the reconstruction
    """

    name = "wavelets"

    def enhance(self, image):

        K = self.params["K"]
        factor = self.params["factor"]
        start_scale = self.params["start_scale"]
        include_background=False
 
        W, I = wavelets(image, scales=K)
    
        W_f = []
        for w in W:
          threshold = factor * (np.std(w))**2
          shrinked = np.maximum((w ** 2 - threshold), 0.0) * 1 / (w+1e-12)
          W_f.append(np.nan_to_num(shrinked, 0))
        
        recon = np.sum(np.stack(W_f[start_scale:]), 0)
        if include_background:
            recon += I[-1]
    
        recon = np.abs(recon)
        return recon 


class LOGDetector(BaseDetector):
    """
    Laplacian of Gaussian based spot enhancing filter

    parameters:
    sigma: sigma used for the Log filter
    """

    name = "LOG"

    def enhance(self, image):

        sigma = self.params["sigma"]

        image = image.astype(np.float32)
        log_response = -scipy.ndimage.gaussian_laplace(image, sigma=sigma)
    
        return log_response


class HessianDetector(BaseDetector):
    """
    Hessian spot enhancing filter

    parameters:
    sigma: sigma used for the Gaussian blurring
    """

    name = "Hessian"

    def enhance(self, image):

        sigma = self.params["sigma"]

        image = image.astype(np.float32)

        J = scipy.ndimage.gaussian_filter(image, sigma)

        Iy, Ix = np.gradient(J)
        Iyy, Iyx = np.gradient(Iy)
        Ixy, Ixx = np.gradient(Ix)

        kappa = Ixx * Iyy - Ixy * Iyx

        C = J * (kappa / kappa.max())

        return C


class TopHatDetector(BaseDetector):
    """
    Grayscale opening Top-Hat filter 

    parameters:
    sigma: sigma used for the Gaussian blurring
    radius: radius of the disk shaped structering element
    """

    name = "TopHat"

    def enhance(self, image):

        sigma = self.params["sigma"]
        radius = self.params["radius"]

        image = image.astype(np.float32)
        J = scipy.ndimage.gaussian_filter(image, sigma)

        selem = skimage.morphology.disk(radius)
        JA = skimage.morphology.opening(J, selem)

        return J - JA


class HDomeDetector(BaseDetector):
    """
    h dome detection scheme 

    parameters:
    h: height of the used h dome
    """

    name = "HDome"

    def enhance(self, image):

        h = self.params["h"]

        image = image.astype(np.float32)

        marker = np.clip(image - h, 0, None)
        reconstructed = skimage.morphology.reconstruction(marker, image, method='dilation')
        hdome = image - reconstructed

        return hdome


class HDomeSmalDetector(BaseDetector):
    """
    h dome detection scheme from step 6 in Quantitative Comparison of Spot Detection
    Methods in Fluorescence Microscopy by Smal et. al. It first takes the LOG transform
    before use the HDome detection

    parameters:
    sigma: sigma used for the LOG filtering
    h: height of the used h dome
    s: exponent used on the output image
    """
    
    name = "HDome-smal"

    def enhance(self, image):

        h = self.params["h"]
        sigma = self.params["sigma"]
        s = self.params["s"]

        image = image.astype(np.float32)
        J = -scipy.ndimage.gaussian_laplace(image, sigma=sigma)

        marker = J - h
        reconstructed = skimage.morphology.reconstruction(marker, J, method='dilation')
        hdome = J - np.clip(reconstructed, 0, None)
        hdome = np.clip(hdome, 0, None)

        return hdome**s


class MPHDDetector(BaseDetector):
    """
    h dome detection scheme from Rezatofighi SH, Hartley R, Hughes WE (2012). A new
    approach for spot detection in total internal reflection
    fluorescence microscopy. In: Proceedings of the 9th
    IEEE Int Symp on Biomedical Imaging. 
    Instead of using a single h value it make a spatially varying h dome based on an initial estimate

    parameters:
    sigma: sigma used for the Gaussian blurring
    h_init: init value of the used h dome
    R: integer radius used to search around local maxima in pixels
    """
    
    name = "MPHD"

    def enhance(self, image):

        h_init = self.params["h_init"]  # h_init: small h for initial maxima detection
        sigma = self.params["sigma"]
        R = self.params["R"]  # R integer search radius (larger than largest object)
        
        image = image.astype(np.float32)
        I = scipy.ndimage.gaussian_filter(image, sigma=sigma)
        
        # Find candidate regional maxima using small h-dome filter
        dome_small = H_dome_transform(I, h=h_init)
        
        # get all local maxima positions
        coords = skimage.feature.peak_local_max(dome_small, threshold_abs=0)
        
        
        # Compute Maximum Possible Height (MPH)
        MA = np.zeros_like(I)
        
        rows, cols = I.shape   
        for (r0, c0) in coords:
        
            center_intensity = I[r0, c0]
            min_boundary_value = np.inf
            optimal_base_pixel = None
        
            # grow circular boundary
            for rad in range(1, R + 1):
        
                boundary_vals = []
        
                for theta in np.linspace(0, 2*np.pi, 36, endpoint=False):
                    rr = int(round(r0 + rad * np.sin(theta)))
                    cc = int(round(c0 + rad * np.cos(theta)))
        
                    if 0 <= rr < rows and 0 <= cc < cols:
                        boundary_vals.append((I[rr, cc], rr, cc))
        
                if not boundary_vals:
                    continue
        
                # maximum intensity on boundary of radius rad
                max_val, rr_max, cc_max = max(boundary_vals, key=lambda x: x[0])
        
                # find radius where this maximum is minimal
                if max_val < min_boundary_value:
                    min_boundary_value = max_val
                    optimal_base_pixel = (rr_max, cc_max)
        
            if optimal_base_pixel is None:
                continue
        
            # MPH = I(Cp) - I(Po)
            Po_val = min_boundary_value
            Hm = center_intensity - Po_val
        
            # Set adaptive mask peak height = I(Po)
            MA[r0, c0] = Po_val
        
        
        # STEP 4: Apply adaptive h-dome filter
        reconstructed = skimage.morphology.reconstruction(MA, I, method='dilation')
        
        return I - reconstructed
        

class SpotiflowDetector(BaseDetector):
    """
    Deep learning based spot detector from:
    Spotiflow: accurate and efficient spot detection for fluorescence microscopy with deep stereographic flow regression

    parameters:
    model_name: selecting which pretrained model to use
    """

    name = "Spotiflow"

    def __init__(self, **params):
        
        from spotiflow.model import Spotiflow
        model_name = params["model_name"]
        self.model = Spotiflow.from_pretrained("general")

    def detect(self, image, threshold, return_segmentation=False):

        points, details = self.model.predict(
            image,
            prob_thresh=threshold,
            subpix=True,
            verbose=False
        )

        enhanced = details.heatmap

        if return_segmentation:
            segmentation = segment_spots(points, enhanced, threshold)
        else: 
            segmentation = None

        return points, enhanced, segmentation

