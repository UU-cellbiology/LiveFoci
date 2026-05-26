from pathlib import Path
import numpy as np
import skimage
import scipy
from lift.general_utils.wavelet_filter import wavelets


#####
#
# helper functions to run detectors on arrays or files
#
#####


def run_detector_stack(stack, detector, threshold, return_segmentation=False):
    """
    Run the specified detector on a numpy array stack (t x h x w).
    Returns detections per frame, the enhanced stack, and label stack.
    """
    detections     = []
    enhanced_stack = []
    label_stack    = []

    for tt, frame in enumerate(stack):
        coords, enhanced, labels = detector.detect(frame, threshold, return_segmentation)
        y, x = coords[:, 0], coords[:, 1]

        if coords.size > 0:
            intensities    = frame[coords[:, 0].astype(int), coords[:, 1].astype(int)]
            coords_with_t  = np.column_stack([x, y, np.full(len(coords), tt), intensities])
        else:
            coords_with_t  = np.empty((0, 4))

        detections.append(coords_with_t)
        enhanced_stack.append(enhanced)
        label_stack.append(labels)

    return detections, np.stack(enhanced_stack), np.stack(label_stack)


def run_detection(path_list, detector, threshold, return_segmentation=False):
    """
    Run the specified detector on a list of .tif stack paths.
    Saves detected_foci.txt (and optionally spot_segmentation.tif) next to each stack.
    """
    for path in path_list:
        path  = Path(path)
        stack = skimage.io.imread(path)

        detections, _, label_stack = run_detector_stack(
            stack, detector, threshold, return_segmentation
        )

        all_detections = np.vstack(detections)
        np.savetxt(path.parent / "detected_foci.txt", all_detections,
                   fmt='%.2f', delimiter='\t')

        if return_segmentation:
            skimage.io.imsave(
                path.parent / "spot_segmentation.tif",
                label_stack.astype(np.uint16),
                check_contrast=False,
            )


#####
#
# detector factory
#
#####


def create_detector(name=None, params=None, threshold=None):
    """
    Create a detector by name, merging provided params with class defaults.
    If name is None, auto-detects the best available method.
    """
    if name is None:
        from lift._helpers import _detect_foci_detector
        name = _detect_foci_detector()

    # registry builds itself from BaseDetector subclasses
    registry = {
        cls.name: cls
        for cls in _all_subclasses(BaseDetector)
        if cls.name != "base"
    }

    if name not in registry:
        raise ValueError(
            f"Unknown detector: {name!r}\n"
            f"Available: {sorted(registry)}"
        )

    cls           = registry[name]
    merged_params = {**cls.default_params, **(params or {})}
    if threshold is None:
        threshold = cls.default_threshold

    print(f"── foci detection method:    {name}")
    print(f"── detection threshold:      {threshold}")
    print(f"── detection params:         {merged_params}")

    return cls(**merged_params), threshold


def _all_subclasses(cls):
    """Recursively collect all subclasses."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


#####
#
# coordinate extraction helpers
#
#####


def H_dome_transform(image, h):
    """Suppress all regional maxima with height less than h."""
    marker        = np.clip(image - h, 0, None)
    reconstructed = skimage.morphology.reconstruction(marker, image, method='dilation')
    return image - reconstructed


def detect_maxima(image, threshold, return_segmentation=False):
    """
    Find local maxima above threshold.
    Optionally returns a watershed segmentation using maxima as markers.
    """
    detections = skimage.feature.peak_local_max(
        image, threshold_abs=threshold, min_distance=1
    )

    labels = segment_spots(detections, image, threshold) if return_segmentation else None
    return detections, labels


def segment_spots(detections, image, threshold):
    """
    Use local maxima as markers in a watershed to segment connected components
    above threshold.
    """
    detected_pixel = np.round(detections).astype(int)

    mask                         = image > threshold
    mask[tuple(detected_pixel.T)] = True

    markers                          = np.zeros(image.shape, dtype=int)
    markers[tuple(detected_pixel.T)] = np.arange(1, len(detected_pixel) + 1)

    return skimage.segmentation.watershed(
        -image, markers=markers, mask=mask, watershed_line=True
    )


def signal_thresholding(image, threshold):
    """
    Segment image by threshold and return connected-component centroids and labels.
    """
    mask       = image > threshold
    labels     = skimage.measure.label(mask, background=0, connectivity=1)
    props      = skimage.measure.regionprops(labels)
    detections = np.array([prop.centroid for prop in props])
    return detections, labels


#####
#
# detector classes
#
#####


class BaseDetector:
    """
    Base class for foci detection methods.
    Subclasses set class-level name, default_params, and default_threshold,
    and implement enhance().
    """
    name              = "base"
    default_params    = {}
    default_threshold = None

    def __init__(self, **params):
        self.params = params

    def enhance(self, image):
        raise NotImplementedError

    def detect(self, image, threshold, return_segmentation=False):
        enhanced              = self.enhance(image)
        coords, segmentation  = detect_maxima(enhanced, threshold, return_segmentation)
        return coords, enhanced, segmentation


class WaveletDetector(BaseDetector):
    """
    Wavelet-based spot enhancing using Jeffreys non-informative prior.

    parameters:
    K           : number of wavelet scales
    factor      : threshold = factor * std(coefficients)^2
    start_scale : first scale included in reconstruction
    """
    name              = "Wavelets"
    default_params    = {"K": 3, "factor": 4.0, "start_scale": 0}
    default_threshold = 30

    def enhance(self, image):
        K           = self.params["K"]
        factor      = self.params["factor"]
        start_scale = self.params["start_scale"]

        W, I = wavelets(image, scales=K)

        W_f = []
        for w in W:
            threshold = factor * (np.std(w)) ** 2
            shrinked  = np.maximum((w ** 2 - threshold), 0.0) / (w + 1e-12)
            W_f.append(np.nan_to_num(shrinked, 0))

        recon = np.abs(np.sum(np.stack(W_f[start_scale:]), 0))
        return recon


class LOGDetector(BaseDetector):
    """
    Laplacian of Gaussian spot enhancing filter.

    parameters:
    sigma : sigma for the LoG filter
    """
    name              = "LOG"
    default_params    = {"sigma": 0.8}
    default_threshold = 26

    def enhance(self, image):
        image = image.astype(np.float32)
        return -scipy.ndimage.gaussian_laplace(image, sigma=self.params["sigma"])


class HessianDetector(BaseDetector):
    """
    Hessian-based spot enhancing filter.

    parameters:
    sigma : sigma for Gaussian blurring before Hessian computation
    """
    name              = "Hessian"
    default_params    = {"sigma": 0.5}
    default_threshold = 5

    def enhance(self, image):
        image = image.astype(np.float32)
        J     = scipy.ndimage.gaussian_filter(image, self.params["sigma"])

        Iy, Ix   = np.gradient(J)
        Iyy, Iyx = np.gradient(Iy)
        Ixy, Ixx = np.gradient(Ix)

        kappa = Ixx * Iyy - Ixy * Iyx
        return J * (kappa / kappa.max())


class TopHatDetector(BaseDetector):
    """
    Grayscale top-hat filter (opening subtraction).

    parameters:
    sigma  : sigma for Gaussian blurring
    radius : radius of the disk-shaped structuring element
    """
    name              = "TopHat"
    default_params    = {"sigma": 0.6, "radius": 2.0}
    default_threshold = 26

    def enhance(self, image):
        image  = image.astype(np.float32)
        J      = scipy.ndimage.gaussian_filter(image, self.params["sigma"])
        selem  = skimage.morphology.disk(self.params["radius"])
        return J - skimage.morphology.opening(J, selem)


class HDomeDetector(BaseDetector):
    """
    H-dome transform for spot detection.

    parameters:
    h : height of the h-dome — regional maxima shorter than h are suppressed
    """
    name              = "HDome"
    default_params    = {"h": 50}
    default_threshold = 40

    def enhance(self, image):
        image = image.astype(np.float32)
        return H_dome_transform(image, h=self.params["h"])


class HDomeSmalDetector(BaseDetector):
    """
    H-dome detection preceded by LoG filtering, from:
    Smal et al., Quantitative Comparison of Spot Detection Methods
    in Fluorescence Microscopy, IEEE TMI 2010.

    parameters:
    sigma : sigma for the LoG pre-filter
    h     : h-dome height
    s     : exponent applied to the output
    """
    name              = "HDome-smal"
    default_params    = {"sigma": 1.2, "h": 80, "s": 1.5}
    default_threshold = 20

    def enhance(self, image):
        h     = self.params["h"]
        sigma = self.params["sigma"]
        s     = self.params["s"]

        image = image.astype(np.float32)
        J     = -scipy.ndimage.gaussian_laplace(image, sigma=sigma)

        hdome = J - np.clip(
            skimage.morphology.reconstruction(J - h, J, method='dilation'), 0, None
        )
        return np.clip(hdome, 0, None) ** s


class MPHDDetector(BaseDetector):
    """
    Maximum Possible Height Dome detector from:
    Rezatofighi et al., A new approach for spot detection in TIRF microscopy,
    ISBI 2012. Computes a spatially adaptive h-dome using local intensity context.

    parameters:
    sigma  : sigma for Gaussian pre-smoothing
    h_init : initial h value for candidate maxima detection
    R      : search radius in pixels for adaptive height estimation
    """
    name              = "MPHD"
    default_params    = {"sigma": 0.5, "h_init": 5, "R": 10}
    default_threshold = 20

    def enhance(self, image):
        h_init = self.params["h_init"]
        sigma  = self.params["sigma"]
        R      = self.params["R"]

        image = image.astype(np.float32)
        I     = scipy.ndimage.gaussian_filter(image, sigma=sigma)

        # candidate maxima from a small h-dome
        dome_small = H_dome_transform(I, h=h_init)
        coords     = skimage.feature.peak_local_max(dome_small, threshold_abs=0)

        rows, cols = I.shape
        MA         = np.zeros_like(I)

        for (r0, c0) in coords:
            min_boundary_value = np.inf

            for rad in range(1, R + 1):
                boundary_vals = []
                for theta in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                    rr = int(round(r0 + rad * np.sin(theta)))
                    cc = int(round(c0 + rad * np.cos(theta)))
                    if 0 <= rr < rows and 0 <= cc < cols:
                        boundary_vals.append(I[rr, cc])

                if not boundary_vals:
                    continue

                max_val = max(boundary_vals)
                if max_val < min_boundary_value:
                    min_boundary_value = max_val

            if min_boundary_value < np.inf:
                MA[r0, c0] = min_boundary_value

        reconstructed = skimage.morphology.reconstruction(MA, I, method='dilation')
        return I - reconstructed


class SpotiflowDetector(BaseDetector):
    """
    Deep learning spot detector from:
    Spotiflow: accurate and efficient spot detection for fluorescence microscopy,
    Dominguez et al., 2024.

    parameters:
    model_name : pretrained model to use (default "general")
    """
    name              = "Spotiflow"
    default_params    = {"model_name": "general"}
    default_threshold = 0.55

    def __init__(self, **params):
        from lift._helpers import _require_spotiflow
        _require_spotiflow()
        from spotiflow.model import Spotiflow
        self.params = params
        self.model  = Spotiflow.from_pretrained(params.get("model_name", "general"))

    def detect(self, image, threshold, return_segmentation=False):
        points, details = self.model.predict(
            image, prob_thresh=threshold, subpix=True, verbose=False
        )
        enhanced    = details.heatmap
        segmentation = segment_spots(points, enhanced, threshold) if return_segmentation else None
        return points, enhanced, segmentation