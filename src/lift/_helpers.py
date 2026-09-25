"""lift._helpers – shared factory functions and dependency guards."""

import importlib


# ── externally-hosted binary bundles (NGMA_utils / NND_utils) ────────────────
#
# These bundle a full JRE + jars each (~150MB) — too large for the pip
# package, so they're hosted on Zenodo instead and downloaded once into a
# local cache on first use.


_ZENODO_HOST = "sandbox.zenodo.org"  
_ZENODO_RECORD_ID = "547985"         
_BINARY_BASE_URL = f"https://{_ZENODO_HOST}/records/{_ZENODO_RECORD_ID}/files"

_BINARY_ARCHIVES = {
    "NGMA_utils": f"{_BINARY_BASE_URL}/NGMA_utils.tar.gz?download=1",
    "NND_utils":  f"{_BINARY_BASE_URL}/NND_utils.tar.gz?download=1",
}


def _ensure_binary_utils(name: str):
    """
    Download and cache one of the externally-hosted binary bundles on first
    use. Returns its local path (a pathlib.Path). Subsequent calls reuse the
    cached copy without re-downloading.

    name: "NGMA_utils" | "NND_utils"
    """
    import tarfile
    import urllib.request
    from pathlib import Path

    if name not in _BINARY_ARCHIVES:
        raise ValueError(
            f"Unknown binary bundle: {name!r}. Available: {list(_BINARY_ARCHIVES)}"
        )

    cache_dir = Path.home() / ".lift-foci" / name
    if cache_dir.exists():
        return cache_dir

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir.parent / f"{name}.tar.gz"
    url = _BINARY_ARCHIVES[name]

    print(f"\n{name} tracker needs its bundled Java tracker — "
          f"downloading once to {cache_dir} ...")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as e:
        raise RuntimeError(
            f"\nCouldn't download {name} from {url}: {e}\n"
            "Check your network connection, or download it manually and "
            f"extract it to {cache_dir}\n"
        ) from None

    print("Download complete, extracting...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(cache_dir.parent)
    archive_path.unlink()

    if not cache_dir.exists():
        raise RuntimeError(
            f"Extraction of {archive_path} didn't produce the expected "
            f"{cache_dir} — check the archive's internal folder structure."
        )

    print(f"{name} ready at {cache_dir}\n")
    return cache_dir


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
    install_name : pip install name if different from importable, e.g. "scikit-image"
    """
    pip_name = install_name or package

    try:
        mod = importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"\n{what} requires '{pip_name}', which is not installed.\n"
            f"Install it with:  pip install \"live-foci[{extra}]\"\n"
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
                f"Upgrade with:  pip install \"live-foci[{extra}]\"\n"
            ) from None
        if max_version and version >= max_version:
            raise ImportError(
                f"\n{what} requires '{pip_name}'<{max_str}, "
                f"but you have {raw}.\n"
                f"Install the correct version with:  pip install \"live-foci[{extra}]\"\n"
            ) from None

    return mod


# ── version probe (no error) ──────────────────────────────────────────────────

def _probe_version(package):
    """Return (major, minor) tuple for an installed package, or None if absent."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        # metadata name may differ from import name e.g. "scikit-image" vs "skimage"
        _METADATA_NAMES = {
            "skimage":   "scikit-image",
            "cv2":       "opencv-python",
        }
        meta_name = _METADATA_NAMES.get(package, package)
        raw = version(meta_name)
        return tuple(int(x) for x in raw.split(".")[:2] if x.isdigit())
    except Exception:
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

# bundled elastix builds, keyed by (sys.platform prefix, machine), relative to
# lift/utils_elastix. The Linux/macOS builds find their shared libs through a
# relative rpath (../lib), so no LD_LIBRARY_PATH / DYLD_LIBRARY_PATH is needed.
_ELASTIX_BINARIES = {
    ("win32",  "amd64"):  "elastix.exe",
    ("linux",  "x86_64"): "elastix_ubuntu/bin/elastix",
    ("darwin", "arm64"):  "elastix_macos/bin/elastix",
}


def _elastix_binary():
    """Return the path to the bundled elastix executable for this platform.

    Elastix registration shells out to a bundled binary via subprocess — it
    needs no extra Python package, so there's nothing to pip-install. Raises RuntimeError on platforms without a bundled build.
    """
    import os
    import platform
    import stat
    import sys
    from pathlib import Path

    system  = sys.platform                     # "win32" | "linux" | "darwin"
    machine = platform.machine().lower()       # "amd64" | "x86_64" | "arm64" | "aarch64"

    rel = _ELASTIX_BINARIES.get((system, machine))
    if rel is None:
        supported = ", ".join(f"{s}/{m}" for s, m in _ELASTIX_BINARIES)
        raise RuntimeError(
            f"\nNo bundled elastix binary for this platform ({system}/{machine}).\n"
            f"Bundled builds: {supported}.\n"
            "Use method='stackreg' instead.\n"
        )

    binary = Path(__file__).resolve().parent / "utils_elastix" / rel
    if not binary.exists():
        raise FileNotFoundError(f"Elastix binary not found: {binary}")

    # wheels built on Windows lose the executable bit, so restore it on first use
    if os.name != "nt" and not os.access(binary, os.X_OK):
        try:
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError as e:
            raise PermissionError(
                f"\nElastix binary is not executable and could not be fixed: {binary}\n"
                f"Run:  chmod +x \"{binary}\"\n"
            ) from e

    return binary


def _has_elastix():
    """True if a bundled elastix binary exists for this platform."""
    try:
        _elastix_binary()
        return True
    except (RuntimeError, OSError):
        return False

def _require_torch():
    return _require(
        "torch", "wavelets", "Wavelet filtering",
        min_version=(1, 9),
        install_name="torch",
    )

# ── auto-detection helpers ────────────────────────────────────────────────────

def _detect_seg_method():
    """Return best available segmentation method, or raise if none installed."""
    v = _probe_version("cellpose")
    if v is None:
        raise ImportError(
            "\nNo segmentation backend is installed.\n"
            "Install one with:\n"
            "  pip install \"live-foci[cp-sam]\"   # cellpose >= 4.0 (recommended)\n"
            "  pip install \"live-foci[cp-v3]\"    # cellpose >= 3.0, < 4.0\n"
        )
    if v >= (4, 0):
        return "cellpose_sam"
    if v >= (3, 0):
        return "cellpose_v3"
    raise ImportError(
        f"\nInstalled cellpose {'.'.join(str(x) for x in v)} is too old.\n"
        "Install a supported version with:\n"
        "  pip install \"live-foci[cp-sam]\"   # cellpose >= 4.0 (recommended)\n"
        "  pip install \"live-foci[cp-v3]\"    # cellpose >= 3.0, < 4.0\n"
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
    """Return best available registration method.

    'elastix' is intentionally never auto-selected here: it's experimental
    and only bundled for some platforms (see _elastix_binary). Users who
    specifically want it must opt in explicitly via parameters.yml
    (step4_registration.method: elastix).
    """
    return "stackreg"   # always available, portable


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