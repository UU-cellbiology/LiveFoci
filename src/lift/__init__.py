"""lift – Live Foci Tracking."""

from lift._helpers import _make_seg_method, _make_preproc


def run(data_path, steps=None):
    """Run full pipeline from parameters.yml. Equivalent to the `lift` CLI command."""
    from lift.cli import run as _run
    return _run(data_path, steps=steps)


def segment(data_path, method=None, min_area=None, preprocessing=None, **kwargs):
    """Segment nuclei in all Pos*/raw folders under data_path.

    Parameters
    ----------
    method : "cellpose_sam" | "cellpose_v3"  (default from config or "cellpose_sam")
    min_area : int  minimum nucleus area in pixels  (default 1000)
    preprocessing : "wavelet_filtering" | "contrast_adjuster" | None
    **kwargs : passed to the segmentation method (e.g. diameter=120, scale_factor=2)
               and/or to the preprocessing function (e.g. scales=10, w_factor=2.2)
    """
    from pathlib import Path
    from lift import nuc_segmentation

    data_path = Path(data_path).resolve()
    cfg      = _step_cfg(data_path, "step1_segmentation")
    seg_cfg  = cfg.get("segmentation",  {})
    prep_cfg = cfg.get("preprocessing", {})

    method = method or seg_cfg.get("method") or None
    min_area      = min_area      or seg_cfg.get("min_area", 1000)
    preprocessing = preprocessing or prep_cfg.get("method")  or None

    method_params = {
        "flow_threshold":     seg_cfg.get("flow_threshold",      0.0),
        "cellprob_threshold": seg_cfg.get("cellprob_threshold", -0.5),
        **seg_cfg.get(method, {}),
        **kwargs,
    }
    preproc_kwargs = prep_cfg.get(preprocessing, {}) if preprocessing else {}

    seg_method   = _make_seg_method(method, method_params)
    preproc_func = _make_preproc(preprocessing)

    raw_paths = sorted(data_path.rglob("Pos*/raw"))
    nuc_segmentation.segment_folderlist(
        raw_paths, seg_method, preproc_func, min_area=min_area, **preproc_kwargs
    )


def track_nuclei(data_path, method=None, min_length=None, **kwargs):
    """Track segmented nuclei using CTC-format masks.

    Parameters
    ----------
    method : "IOU" | "NND" | "trackastra"  (default from config or "IOU")
    min_length : int  minimum track length in frames  (default 20)
    **kwargs : tracker-specific options, e.g. iou_min=0.05, max_distance=30,
               gap_closing=2, remove_gaps=True
    """
    from pathlib import Path
    from lift import nuclei_trackers

    data_path = Path(data_path).resolve()
    cfg = _step_cfg(data_path, "step2_tracking")

    method = method or cfg.get("method") or None
    min_length = min_length or cfg.get("min_length", 20)
    kw         = {**cfg.get(method, {}), **kwargs}

    seg_paths = sorted(data_path.rglob("Pos*/results/result_cell_seg"))
    nuclei_trackers.run_tracker(seg_paths, method=method, min_length=min_length, **kw)


def crop(data_path, margin=None):
    """Cut individual cells from tracked sequences.

    Parameters
    ----------
    margin : int  padding around the cell bounding box in pixels  (default 30)
    """
    from pathlib import Path
    from lift import cut_out_cells

    data_path = Path(data_path).resolve()
    cfg    = _step_cfg(data_path, "step3_cropping")
    margin = margin if margin is not None else cfg.get("margin", 30)

    for pp in sorted(data_path.rglob("Pos*")):
        try:
            cut_out_cells.cut_from_ctc(pp, margin=margin)
        except ValueError:
            pass


def register(data_path, method=None, preprocessing=None, **kwargs):
    """Register cropped cell stacks to correct for motion.

    Parameters
    ----------
    method : "stackreg" | "elastix"  (default from config or "stackreg")
    preprocessing : "wavelet_denoise" | "DOG_filter" | "threshold" | None
    **kwargs : passed to the registration method (elastix options, etc.)
    """
    from pathlib import Path
    from lift import registration

    data_path = Path(data_path).resolve()
    cfg = _step_cfg(data_path, "step4_registration")

    method = method or cfg.get("method") or None
    preprocessing = preprocessing or cfg.get("preprocessing") or None
    kw            = {**cfg.get("elastix", {}), **kwargs} if method == "elastix" else kwargs

    preproc_func = _make_preproc(preprocessing)
    followed     = sorted(data_path.rglob("Pos*/results/followed/*/I_*.tif"))
    registration.run_registration(
        followed, method=method, preprocess_function=preproc_func, **kw
    )


def detect(data_path, method=None, threshold=None, return_segmentation=None, **kwargs):
    """Detect foci in registered cell stacks.

    Parameters
    ----------
    method : "Wavelets" | "LOG" | "Hessian" | "TopHat" | "HDome" |
             "HDome-smal" | "MPHD" | "Spotiflow"
    threshold : float  detection threshold (uses per-method default if omitted)
    return_segmentation : bool  also save spot_segmentation.tif  (default False)
    **kwargs : detector parameters, e.g. K=3, factor=4.0, sigma=0.8
    """
    from pathlib import Path
    from lift import foci_detection

    data_path = Path(data_path).resolve()
    cfg = _step_cfg(data_path, "step5_detection")

    method              = method    or cfg.get("method")
    threshold           = threshold if threshold is not None else cfg.get("threshold")
    return_segmentation = return_segmentation if return_segmentation is not None \
                          else cfg.get("return_segmentation", False)
    method_params       = {**cfg.get("params", {}), **kwargs}

    if method is None:
        raise ValueError(
            "detect() requires a method. "
            "Pass method= or set step5_detection.method in parameters.yml"
        )

    detector, threshold = foci_detection.create_detector(
        method, params=method_params, threshold=threshold
    )
    registered = sorted(data_path.rglob("Pos*/results/registered/*/I_*.tif"))
    foci_detection.run_detection(
        registered, detector, threshold, return_segmentation=return_segmentation
    )


def track_foci(data_path, method=None, min_track_length=None, **kwargs):
    """Link detected foci across frames into trajectories.

    Parameters
    ----------
    method : "GNN" | "NGMA" | "trackastra"
    min_track_length : int  minimum number of detections per track  (default 3)
    **kwargs : tracker-specific options, e.g. max_distance=5, gap_closing=2,
               use_segmentation=True
    """
    from pathlib import Path
    from lift import foci_trackers

    data_path = Path(data_path).resolve()
    cfg = _step_cfg(data_path, "step6_tracking")

    method           = method           or cfg.get("method")
    min_track_length = min_track_length if min_track_length is not None \
                       else cfg.get("min_track_length", 3)
    kw = {**cfg.get(method, {}), **kwargs} if method else kwargs

    if method is None:
        raise ValueError(
            "track_foci() requires a method. "
            "Pass method= or set step6_tracking.method in parameters.yml"
        )

    registered = sorted(data_path.rglob("Pos*/results/registered/*/I_*.tif"))
    foci_trackers.run_foci_tracker(
        registered, method=method, min_track_length=min_track_length, **kw
    )


def _step_cfg(data_path, step_key: str) -> dict:
    from lift.general_utils.params_utils import load_params
    p = load_params(data_path) or {}
    return p.get(step_key, {})


__all__ = [
    "run",
    "segment",
    "track_nuclei",
    "crop",
    "register",
    "detect",
    "track_foci",
]