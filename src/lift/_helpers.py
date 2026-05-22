"""lift._helpers – shared factory functions and dependency guards."""

import importlib


def _require(package, extra, what, min_version=None, max_version=None):
    """
    Import *package* and optionally check its version.

    Parameters
    ----------
    package     : importable name, e.g. "cellpose"
    extra       : pip extra name, e.g. "cp-sam"
    what        : human description, e.g. "Cellpose-SAM segmentation"
    min_version : tuple of ints, e.g. (4, 0) — inclusive lower bound
    max_version : tuple of ints, e.g. (5, 0) — exclusive upper bound
    """
    try:
        mod = importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"\n{what} requires '{package}', which is not installed.\n"
            f"Install it with:  pip install \"LiFT[{extra}]\"\n"
        ) from None

    if min_version is not None or max_version is not None:
        raw = getattr(mod, "__version__", None)
        if raw is None:
            return mod  # can't check, proceed

        # strip any suffixes like "4.0.1.dev0" → (4, 0, 1)
        try:
            version = tuple(int(x) for x in raw.split(".")[:3] if x.isdigit())
        except Exception:
            return mod  # unparseable version, proceed

        min_str = ".".join(str(x) for x in min_version) if min_version else None
        max_str = ".".join(str(x) for x in max_version) if max_version else None

        too_old = min_version and version < min_version
        too_new = max_version and version >= max_version

        if too_old and too_new:  # shouldn't happen but be safe
            raise ImportError(
                f"\n{what} requires '{package}' >={min_str}, <{max_str}, "
                f"but you have {raw}.\n"
                f"Install it with:  pip install \"LiFT[{extra}]\"\n"
            )
        if too_old:
            raise ImportError(
                f"\n{what} requires '{package}'>={min_str}, "
                f"but you have {raw}.\n"
                f"Upgrade with:  pip install \"LiFT[{extra}]\"\n"
            ) from None
        if too_new:
            raise ImportError(
                f"\n{what} requires '{package}'<{max_str}, "
                f"but you have {raw}.\n"
                f"Install the correct version with:  pip install \"LiFT[{extra}]\"\n"
            ) from None

    return mod


# ── per-dependency convenience wrappers ──────────────────────────────────────

def _require_cellpose_sam():
    return _require(
        "cellpose", "cp-sam", "Cellpose-SAM segmentation",
        min_version=(4, 0),
    )

def _require_cellpose_v3():
    return _require(
        "cellpose", "cp-v3", "Cellpose-v3 segmentation",
        min_version=(3, 0), max_version=(4, 0),
    )

def _require_trackastra():
    return _require(
        "trackastra", "trackastra", "Trackastra tracking",
        min_version=(0, 2),
    )

def _require_spotiflow():
    return _require(
        "spotiflow", "spotiflow", "Spotiflow foci detection",
        min_version=(0, 4),
    )

def _require_elastix():
    return _require(
        "itk", "elastix", "Elastix registration",
        min_version=(5, 3),
    )


# ── segmentation method factory ───────────────────────────────────────────────

def _make_seg_method(name, params):
    if name == 'cellpose_sam':
        _require_cellpose_sam()
        from lift.nuc_segmentation import CP_SAM
        return CP_SAM(
            flow_threshold     = params.get('flow_threshold',      0.0),
            cellprob_threshold = params.get('cellprob_threshold', -0.5),
            scale_factor       = params.get('scale_factor',          1),
        )
    if name == 'cellpose_v3':
        _require_cellpose_v3()
        from lift.nuc_segmentation import CP_V3
        return CP_V3(
            flow_threshold     = params.get('flow_threshold',      0.0),
            cellprob_threshold = params.get('cellprob_threshold', -0.5),
            diameter           = params.get('diameter',           140),
        )
    raise ValueError(f"Unknown segmentation method: {name!r}")


# ── preprocessing function factory ───────────────────────────────────────────

def _make_preproc(name):
    if name in (None, 'None', ''):
        return None
    mapping = {
        'wavelet_denoise':   ('lift.registration',     'wavelet_denoise'),
        'threshold':         ('lift.registration',     'threshold'),
        'DOG_filter':        ('lift.registration',     'DOG_filter'),
        'wavelet_filtering': ('lift.nuc_segmentation', 'wavelet_filtering'),
        'contrast_adjuster': ('lift.nuc_segmentation', 'contrast_adjuster'),
    }
    if name not in mapping:
        raise ValueError(f"Unknown preprocessing function: {name!r}")
    module_path, func_name = mapping[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)