from pathlib import Path
import tempfile
import numpy as np
import skimage
from pystackreg import StackReg
from pystackreg.util import to_uint16
from lift.general_utils.wavelet_filter import wavelets


#####
#
# below are the helper functions for running all different registration options and saving the output
#
#####


def register_array(stack, method, preprocess_function=None, **kwargs):
    """
    Perform registration on a single stack using the specified method.

    stack: numpy array of shape (t, height, width)
    method: "stackreg" | "elastix"
    preprocess_function: optional function applied before computing transforms
    """
    if method == 'stackreg':
        return stackreg_registration(stack, preprocess_function)
    elif method == 'elastix':
        loss = kwargs.pop("loss", "MSE")
        return ElastixReg(loss, preprocess_function).register_stack(stack)
    else:
        raise ValueError(f"Unknown registration method: {method!r}")


def run_registration(path_list, method, preprocess_function=None, save_name='registered', **kwargs):
    """
    Run registration on a list of .tif stack paths and save results.

    path_list: list of paths to cropped cell stacks (I_*.tif)
    method: "stackreg" | "elastix"
    preprocess_function: optional preprocessing applied before computing transforms
    save_name: output subfolder name (default "registered")
    """
    if preprocess_function:
        print(f"running {method} registration with {preprocess_function.__name__} as preprocessing")
    else:
        print(f"running {method} registration without preprocessing")

    for path in path_list:
        path = Path(path)
        stack = skimage.io.imread(path)

        registered_stack = register_array(stack, method, preprocess_function, **kwargs)

        tifname    = path.name
        nucleus_id = path.parent.name
        pos_dir    = path.parents[2]
        save_dir   = pos_dir / save_name / nucleus_id
        save_dir.mkdir(parents=True, exist_ok=True)

        skimage.io.imsave(save_dir / tifname, registered_stack, check_contrast=False)


#####
#
# below are the different registration methods
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

    registered_stack = np.clip(to_uint16(registered_stack), 0, 255).astype(np.uint8)
    return registered_stack


class ElastixReg:
    """
    ITK-Elastix based image registration — cross-platform pure-Python replacement
    for the old elastix.exe subprocess approach.

    Install the optional dependency with:
        pip install "LiFT[elastix]"

    Parameters
    ----------
    loss : "MSE" | "MI" | "NCC"
        Similarity metric. Selects the matching parameter file from elastix_params/.
    preprocess_function : callable | None
        Applied to the full stack before computing transforms. The raw stack
        is always used for the actual transformation.
    previous_initialisation : bool
        Initialise each frame's registration from the previous transform.
    """

    # parameter files shipped with the package
    _PARAMS_DIR = Path(__file__).resolve().parent / "elastix_params"

    def __init__(self, loss='MSE', preprocess_function=None, previous_initialisation=True):
        try:
            import itk
            self._itk = itk
        except ImportError:
            raise ImportError(
                "itk-elastix is required for elastix registration. "
                "Install it with: pip install \"LiFT[elastix]\""
            )

        self.loss             = loss
        self.preproc_function = preprocess_function
        self.prev_init        = previous_initialisation

        self._param_file = self._PARAMS_DIR / (
            f"elastix_parameters_MultiImage_{loss}.txt"
            if preprocess_function
            else f"elastix_parameters_{loss}.txt"
        )

        if not self._param_file.exists():
            raise FileNotFoundError(
                f"Elastix parameter file not found: {self._param_file}\n"
                f"Available files: {list(self._PARAMS_DIR.glob('*.txt'))}"
            )

    def _to_itk(self, arr):
        """Convert a 2-D numpy array to an ITK image."""
        return self._itk.GetImageFromArray(arr.astype(np.float32))

    def _register_pair(self, fixed_arr, moving_arr,
                       fixed_proc=None, moving_proc=None,
                       initial_transform=None):
        """
        Register one moving frame to one fixed frame.
        Returns (registered_array, result_transform).
        """
        itk = self._itk

        fixed_img  = self._to_itk(fixed_arr)
        moving_img = self._to_itk(moving_arr)

        param_obj = itk.ParameterObject.New()
        param_obj.ReadParameterFile(str(self._param_file))

        reg = itk.ElastixRegistrationMethod.New(fixed_img, moving_img)
        reg.SetParameterObject(param_obj)

        if self.preproc_function and fixed_proc is not None:
            reg.SetFixedImage(1,  self._to_itk(fixed_proc))
            reg.SetMovingImage(1, self._to_itk(moving_proc))

        if initial_transform is not None and self.prev_init:
            reg.SetInitialTransformParameterObject(initial_transform)

        reg.SetLogToConsole(False)
        reg.Update()

        result      = np.array(itk.GetArrayFromImage(reg.GetOutput()))
        transform   = reg.GetTransformParameterObject()
        return result, transform

    def register_stack(self, img_stack):
        """
        Register all frames in img_stack to the previous frame.

        Returns a uint8 array of the same shape as img_stack.
        """
        processed_stack = self.preproc_function(img_stack) if self.preproc_function else None

        registered = [img_stack[0].copy()]
        prev_transform = None

        for tt in range(len(img_stack) - 1):
            fixed_arr  = registered[-1]   # always register to the last registered frame
            moving_arr = img_stack[tt + 1]

            fixed_proc  = (self.preproc_function([fixed_arr])[0]
                           if self.preproc_function else None)
            moving_proc = (processed_stack[tt + 1]
                           if processed_stack is not None else None)

            result, prev_transform = self._register_pair(
                fixed_arr, moving_arr,
                fixed_proc=fixed_proc,
                moving_proc=moving_proc,
                initial_transform=prev_transform if self.prev_init else None,
            )

            registered.append(result)

        registered = np.stack(registered)
        return np.clip(registered, 0, 255).astype(np.uint8)


#####
#
# preprocessing functions for registration
#
#####


def wavelet_denoise(stack):
    num_scales  = 2
    factor      = 5.0
    start_scale = 1

    preprocessed = []
    for frame in stack:
        W, I = wavelets(frame, scales=num_scales)
        for w in W:
            threshold = factor * (np.std(w)) ** 2
            w[w ** 2 < threshold] = 0
        coefs      = np.sum(np.stack(W[start_scale:]), 0)
        recon_full = np.clip(coefs, 0, None)
        preprocessed.append(recon_full)

    return np.stack(preprocessed)


def DOG_filter(stack):
    sig_low  = 0.75
    sig_high = 1.0

    preprocessed = []
    for frame in stack:
        processed = skimage.filters.difference_of_gaussians(
            frame.astype(np.float32), low_sigma=sig_low, high_sigma=sig_high
        )
        preprocessed.append(processed)

    return np.stack(preprocessed)


def threshold(stack):
    rel_threshold = 0.1

    preprocessed = []
    for frame in stack:
        processed = frame.copy()
        processed[frame < processed.max() * rel_threshold] = 0
        preprocessed.append(processed)

    return np.stack(preprocessed)