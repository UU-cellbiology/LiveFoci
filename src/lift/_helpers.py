"""lift._helpers – shared factory functions and dependency guards."""

import importlib
import os
import platform
import stat
import sys
from pathlib import Path


# ── externally-hosted binary bundles ──────────────────────────────
#
# Too large for the pip package, so they are hosted on Zenodo and downloaded
# once into a local cache on first use.

_ZENODO_HOST = "zenodo.org"
_ZENODO_RECORD_ID = "23417069"   # update whenever a new Zenodo version is published
_BINARY_BASE_URL = f"https://{_ZENODO_HOST}/records/{_ZENODO_RECORD_ID}/files"

# name -> (archive file name on Zenodo, expected SHA-256).
# Every archive must contain a single top-level folder called <name>.
# A SHA-256 of None skips verification.
# Get the elastix hashes by running make_elastix_archives.py.
_BINARY_ARCHIVES = {
    "NGMA_utils":      ("NGMA_utils.tar.gz",   None),
    "NND_utils":       ("NND_utils.tar.gz",    None),
    "elastix_windows": ("elastix_windows.zip", "PASTE_SHA256"),
    "elastix_ubuntu":  ("elastix_ubuntu.zip",  "PASTE_SHA256"),
    "elastix_macos":   ("elastix_macos.zip",   "PASTE_SHA256"),
}


def _cache_root():
    """Cache folder for downloaded bundles (override with LIFT_CACHE_DIR)."""
    return Path(os.environ.get("LIFT_CACHE_DIR") or Path.home() / ".lift-foci")


def _ensure_binary_utils(name: str):
    """
    Download and cache one of the externally-hosted bundles on first use.
    Returns its local path (a pathlib.Path). Subsequent calls reuse the cached
    copy without re-downloading.

    name: a key of _BINARY_ARCHIVES, e.g. "NGMA_utils" or "elastix_windows"
    """
    import hashlib
    import tarfile
    import tempfile
    import urllib.request
    import zipfile

    if name not in _BINARY_ARCHIVES:
        raise ValueError(
            f"Unknown binary bundle: {name!r}. Available: {list(_BINARY_ARCHIVES)}"
        )

    cache_dir = _cache_root() / name
    if cache_dir.exists():
        return cache_dir

    filename, expected_sha256 = _BINARY_ARCHIVES[name]
    url = f"{_BINARY_BASE_URL}/{filename}?download=1"
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{name} is not installed locally — "
          f"downloading once to {cache_dir} ...")

    # Work in a temp folder next to the cache, so a failed or interrupted
    # download never leaves a half-extracted cache_dir behind.
    with tempfile.TemporaryDirectory(dir=cache_dir.parent, prefix=f".{name}-") as tmp:
        tmp = Path(tmp)
        archive_path = tmp / filename

        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as e:
            raise RuntimeError(
                f"\nCouldn't download {name} from {url}: {e}\n"
                "Check your network connection, or download it manually and "
                f"extract it to {cache_dir}\n"
            ) from None

        if expected_sha256 is not None:
            digest = hashlib.sha256()
            with open(archive_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256.lower():
                raise RuntimeError(
                    f"\nChecksum mismatch for {filename} downloaded from {url}\n"
                    f"  expected {expected_sha256}\n"
                    f"  got      {digest.hexdigest()}\n"
                )

        print("Download complete, extracting...")
        extract_dir = tmp / "extracted"
        extract_dir.mkdir()
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)
        else:
            with tarfile.open(archive_path) as tf:
                # the "data" filter blocks paths that escape the target folder
                kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                tf.extractall(extract_dir, **kwargs)

        extracted = extract_dir / name
        if not extracted.is_dir():
            raise RuntimeError(
                f"Extraction of {filename} didn't produce the expected "
                f"top-level folder {name!r} — check the archive's internal "
                "folder structure."
            )

        try:
            extracted.rename(cache_dir)
        except OSError:
            # fine if a parallel process finished first, otherwise a real error
            if not cache_dir.exists():
                raise

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

# ── elastix ───────────────────────────────────────────────────────
#
# Which bundle (a key of _BINARY_ARCHIVES) to use per platform, keyed by
# (sys.platform, machine). Inside every bundle the executables live in bin/.
# The Linux/macOS builds find their shared libs through a relative rpath
# (../lib); the Windows build keeps its DLL next to the executables in bin/,
# where Windows looks for it.
_ELASTIX_BINARIES = {
    ("win32",  "amd64"):  "elastix_windows",
    ("linux",  "x86_64"): "elastix_ubuntu",
    ("darwin", "arm64"):  "elastix_macos",
}


def _make_executable(path):
    """Restore the executable bit (zips built on Windows don't carry it)."""
    if os.name == "nt" or not path.exists() or os.access(path, os.X_OK):
        return
    try:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError as e:
        raise PermissionError(
            f"\nBinary is not executable and could not be fixed: {path}\n"
            f"Run:  chmod +x \"{path}\"\n"
        ) from e


def _elastix_binary():
    """Return the path to the elastix executable for this platform.

    On first use the matching build is downloaded from Zenodo into the cache
    (see _ensure_binary_utils). To use your own install instead, set
    LIFT_ELASTIX_DIR to a folder containing elastix and transformix.
    """
    exe_suffix = ".exe" if os.name == "nt" else ""

    override = os.environ.get("LIFT_ELASTIX_DIR")
    if override:
        binary = Path(override) / f"elastix{exe_suffix}"
        if not binary.exists():
            raise FileNotFoundError(
                f"LIFT_ELASTIX_DIR is set, but {binary} does not exist"
            )
    else:
        system  = sys.platform                  # "win32" | "linux" | "darwin"
        machine = platform.machine().lower()    # "amd64" | "x86_64" | "arm64" | ...

        bundle = _ELASTIX_BINARIES.get((system, machine))
        if bundle is None:
            supported = ", ".join(f"{s}/{m}" for s, m in _ELASTIX_BINARIES)
            raise RuntimeError(
                f"\nNo elastix build for this platform ({system}/{machine}).\n"
                f"Available builds: {supported}.\n"
                "Use method='stackreg' instead, or set LIFT_ELASTIX_DIR to a "
                "folder containing your own elastix and transformix executables.\n"
            )

        binary = _ensure_binary_utils(bundle) / "bin" / f"elastix{exe_suffix}"
        if not binary.exists():
            raise FileNotFoundError(f"Elastix binary not found: {binary}")

    _make_executable(binary)
    _make_executable(binary.with_name(f"transformix{exe_suffix}"))
    return binary


def _has_elastix():
    """True if elastix can be used on this platform.

    Deliberately does NOT download anything: the download happens on the
    first call to _elastix_binary().
    """
    if os.environ.get("LIFT_ELASTIX_DIR"):
        return True
    return (sys.platform, platform.machine().lower()) in _ELASTIX_BINARIES

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