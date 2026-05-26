"""lift._helpers – shared factory functions and dependency guards."""

import importlib


# ── core guard ────────────────────────────────────────────────────────────────

def _require(package, extra, what, min_version=None, max_version=None, install_name=None):
    """
    Import *package* and optionally check its version.

    Parameters
    ----------
    package      : importable name, e.g. "cellpose"
    extra        : pip extra name, e.g. "cp-sam"
    what         : human description, e.g. "Cellpose-SAM segmentation"
    min_version  : tuple of ints, e.g. (4, 0) — inclusive lower bound
    max_version  : tuple of ints, e.g. (5, 0) — exclusive upper bound
    install_name : pip install name if different from importable, e.g. "itk-elastix"
    """
    pip_name = install_name or package

    try:
        mod = importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"\n{what} requires '{pip_name}', which is not installed.\n"
            f"Install it with:  pip install \"LiFT[{extra}]\"\n"
        ) from None

    if min_version is not None or max_version is not None:
        raw = getattr(mod, "__version__", None)
        if raw is None:
            return mod

        try:
            version = tuple(int(x) for x in raw.split(".")[:3] if x.isdigit())
        except Exception:
            return mod

        min_str = ".".join(str(x) for x in min_version) if min_version else None
        max_str = ".".join(str(x) for x in max_version) if max_version else None

        if min_version and version < min_version:
            raise ImportError(
                f"\n{what} requires '{pip_name}'>={min_str}, "
                f"but you have {raw}.\n"
                f"Upgrade with:  pip install \"LiFT[{extra}]\"\n"
            ) from None
        if max_version and version >= max_version:
            raise ImportError(
                f"\n{what} requires '{pip_name}'<{max_str}, "
                f"but you have {raw}.\n"
                f"Install the correct version with:  pip install \"LiFT[{extra}]\"\n"
            ) from None

    return mod


# ── version probe (no error) ──────────────────────────────────────────────────

def _probe_version(package):
    """Return (major, minor) tuple for an installed package, or None if absent."""
    try:
        mod = importlib.import_module(package)
        raw = getattr(mod, "__version__", "")
        return tuple(int(x) for x in raw.split(".")[:2] if x.isdigit())
    except ImportError:
        return None


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
        install_name="itk-elastix",
    )


# ── auto-detection helpers ────────────────────────────────────────────────────

def _detect_seg_method():
    """Return best available segmentation method, or raise if none installed."""
    v = _probe_version("cellpose")
    if v is None:
        raise ImportError(
            "\nNo segmentation backend is installed.\n"
            "Install one with:\n"
            "  pip install \"LiFT[cp-sam]\"   # cellpose >= 4.0 (recommended)\n"
            "  pip install \"LiFT[cp-v3]\"    # cellpose >= 3.0, < 4.0\n"
        )
    if v >= (4, 0):
        return "cellpose_sam"
    if v >= (3, 0):
        return "cellpose_v3"
    raise ImportError(
        f"\nInstalled cellpose {'.'.join(str(x) for x in v)} is too old.\n"
        "Install a supported version with:\n"
        "  pip install \"LiFT[cp-sam]\"   # cellpose >= 4.0 (recommended)\n"
        "  pip install \"LiFT[cp-v3]\"    # cellpose >= 3.0, < 4.0\n"
    )


def _detect_nuclei_tracker():
    """Return best available nuclei tracking method."""
    if _probe_version("trackastra") is not None:
        return "trackastra"
    return "IOU"   # always available, no optional dep


def _detect_foci_tracker():
    """Return best available foci tracking method."""
    if _probe_version("trackastra") is not None:
        return "trackastra"
    return "GNN"   # always available, no optional dep


def _detect_foci_detector():
    """Return best available foci detection method."""
    if _probe_version("spotiflow") is not None:
        return "Spotiflow"
    return "Wavelets"   # always available, no optional dep


def _detect_registration_method():
    """Return best available registration method."""
    if _probe_version("itk") is not None:
        return "elastix"
    return "stackreg"   # always available, no optional dep


# ── segmentation factory ──────────────────────────────────────────────────────

def _make_seg_method(name, params):
    if name is None:
        name = _detect_seg_method()

    print(f"segmentation method:      {name}")

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
    raise ValueError(
        f"Unknown segmentation method: {name!r}\n"
        f"Available: 'cellpose_sam', 'cellpose_v3'"
    )


# ── preprocessing factory ─────────────────────────────────────────────────────

def _make_preproc(name):
    if name in (None, 'None', ''):
        print(f"preprocessing:            none\n")
        return None

    mapping = {
        'wavelet_denoise':   ('lift.registration',     'wavelet_denoise'),
        'threshold':         ('lift.registration',     'threshold'),
        'DOG_filter':        ('lift.registration',     'DOG_filter'),
        'wavelet_filtering': ('lift.nuc_segmentation', 'wavelet_filtering'),
        'contrast_adjuster': ('lift.nuc_segmentation', 'contrast_adjuster'),
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown preprocessing function: {name!r}\n"
            f"Available: {list(mapping)}"
        )

    print(f"preprocessing:            {name}\n")
    module_path, func_name = mapping[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)