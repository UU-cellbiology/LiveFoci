"""lift.general_utils – shared helper functions, re-exported for convenience.

`wavelets` is exposed lazily (PEP 562): wavelet_filter.py imports torch
at module level, and nuc_segmentation.py / foci_detection.py / registration.py
all do `from lift.general_utils import wavelet_filter, ...` or
`from lift.general_utils.wavelet_filter import wavelets` at their own top
level. Keeping it lazy here means `import lift.general_utils` alone
doesn't force torch to load.
"""

from lift.general_utils.params_utils import load_params, save_section, yaml_path
from lift.general_utils.load_sequence import load

__all__ = [
    "load_params", "save_section", "yaml_path",
    "load", "extract_tracking_info", "wavelets",
]


def __getattr__(name):
    if name == "wavelets":
        from lift.general_utils.wavelet_filter import wavelets
        return wavelets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")