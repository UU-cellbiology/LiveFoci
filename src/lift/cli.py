"""lift.cli – command-line entry point and full pipeline runner."""

from pathlib import Path
from lift._helpers import _make_seg_method, _make_preproc
from lift.general_utils.params_utils import load_params


def run(data_path: str, steps: list = None):
    """Run pipeline steps defined in parameters.yml inside data_path.

    steps: list of ints (1-6) to execute; None runs all present sections.
    """
    data_path = Path(data_path).resolve()
    p = load_params(data_path)

    if not p:
        print(f"No parameters.yml found in {data_path}. Nothing to run.")
        return

    def _should_run(step_num):
        return steps is None or step_num in steps

    # ── step 1: nuclei segmentation ──────────────────────────────────────────
    if _should_run(1) and (s := p.get('step1_segmentation')):
        print("── Step 1: Nuclei Segmentation")
        from lift import nuc_segmentation

        seg          = s.get('segmentation',  {})
        prepr        = s.get('preprocessing', {})
        method_name  = seg.get('method', 'cellpose_sam')
        preproc_name = prepr.get('method')
        min_area     = seg.get('min_area', 1000)
        method_params = {
            'flow_threshold':     seg.get('flow_threshold',      0.0),
            'cellprob_threshold': seg.get('cellprob_threshold', -0.5),
            **seg.get(method_name, {}),
        }

        method         = _make_seg_method(method_name, method_params)
        preproc_func   = _make_preproc(preproc_name)
        preproc_kwargs = prepr.get(preproc_name, {}) if preproc_name not in (None, 'None', '') else {}
        raw_paths = sorted(data_path.rglob("Pos*/raw"))        
        nuc_segmentation.segment_folderlist(
            raw_paths, method, preproc_func, min_area=min_area, **preproc_kwargs
        )

    # ── step 2: nuclei tracking ──────────────────────────────────────────────
    if _should_run(2) and (s := p.get('step2_tracking')):
        print("── Step 2: Nuclei Tracking")
        from lift import nuclei_trackers

        method        = s.get('method') or None
        min_length    = s.get('min_length', 20)
        method_params = s.get(method, {})

        seg_paths = sorted(data_path.rglob("Pos*/results/result_cell_seg"))
        nuclei_trackers.run_tracker(
            seg_paths, method=method, min_length=min_length, **method_params
        )

    # ── step 3: cell cropping ────────────────────────────────────────────────
    if _should_run(3) and (s := p.get('step3_cropping')):
        print("── Step 3: Cell Cropping\n")
        from lift import cut_out_cells

        for pp in sorted(data_path.rglob("Pos*")):
            try:
                cut_out_cells.cut_from_ctc(pp, margin=s.get('margin', 30))
            except ValueError:
                pass

    # ── step 4: registration ─────────────────────────────────────────────────
    if _should_run(4) and (s := p.get('step4_registration')):
        print("── Step 4: Registration")
        from lift import registration

        method = s.get('method') or None
        preproc_func = _make_preproc(s.get('preprocessing'))
        kwargs       = s.get('elastix', {}) if method == 'elastix' else {}

        followed = sorted(data_path.rglob("Pos*/results/followed/*/I_*.tif"))
        registration.run_registration(
            followed, method=method, preprocess_function=preproc_func, **kwargs
        )

    # ── step 5: foci detection ───────────────────────────────────────────────
    if _should_run(5) and (s := p.get('step5_detection')):
        print("── Step 5: Foci Detection\n")
        from lift import foci_detection

        s = dict(s)
        method_params    = s.pop('params', {})
        method           = s.pop('method', None)
        threshold        = s.pop('threshold', None)
        return_seg       = s.pop('return_segmentation', False)

        detector, threshold = foci_detection.create_detector(
            method, params=method_params, threshold=threshold
        )

        registered = sorted(data_path.rglob("Pos*/results/registered/*/I_*.tif"))
        foci_detection.run_detection(
            registered, detector, threshold, return_segmentation=return_seg
        )

    # ── step 6: foci tracking ────────────────────────────────────────────────
    if _should_run(6) and (s := p.get('step6_tracking')):
        print("── Step 6: Foci Tracking")
        from lift import foci_trackers

        method           = s.get('method') or None
        min_track_length = s.get('min_track_length', 3)
        method_params    = s.get(method, {})

        registered = sorted(data_path.rglob("Pos*/results/registered/*/I_*.tif"))
        foci_trackers.run_foci_tracker(
            registered, method=method, min_track_length=min_track_length, **method_params
        )

    print("Done.")

def init(data_path: str):
    """
    Generate a parameters.yml in data_path based on what is installed.
    Methods requiring uninstalled packages are noted in comments.
    """
    from lift._helpers import _probe_version
    from lift.general_utils.params_utils import yaml_path, save_section
    from pathlib import Path
    import yaml

    data_path = Path(data_path).resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    out = yaml_path(data_path)
    if out.exists():
        print(f"parameters.yml already exists at {out} — skipping.")
        return

    # ── probe installed packages ──────────────────────────────────────────────
    cellpose_v = _probe_version("cellpose")
    has_cp_sam      = cellpose_v is not None and cellpose_v >= (4, 0)
    has_cp_v3       = cellpose_v is not None and (3, 0) <= cellpose_v < (4, 0)
    has_trackastra  = _probe_version("trackastra") is not None
    has_spotiflow   = _probe_version("spotiflow")  is not None
    has_elastix     = _probe_version("itk")        is not None

    # ── pick best available defaults ──────────────────────────────────────────
    if has_cp_sam:
        seg_method = "cellpose_sam"
    elif has_cp_v3:
        seg_method = "cellpose_v3"
    else:
        seg_method = None

    nuc_tracker   = "trackastra" if has_trackastra else "IOU"
    foci_detector = "Spotiflow"  if has_spotiflow  else "Wavelets"
    foci_tracker  = "trackastra" if has_trackastra else "GNN"
    reg_method    = "elastix"    if has_elastix    else "stackreg"

    # ── build the config dict ─────────────────────────────────────────────────
    config = {}

    # step 1
    if seg_method:
        config["step1_segmentation"] = {
            "segmentation": {
                "method":             seg_method,
                "flow_threshold":     0.0,
                "cellprob_threshold": -0.5,
                "min_area":           1000,
                "cellpose_sam":       {"scale_factor": 1},
                "cellpose_v3":        {"diameter": 140},
            },
            "preprocessing": {
                "method": None,
                "wavelet_filtering": {"scales": 5, "w_factor": 1.5, "start_scale": 2},
                "contrast_adjuster": {"sigma": 3.0, "c_factor": 6.0},
            },
        }

    # step 2
    config["step2_tracking"] = {
        "method":     nuc_tracker,
        "min_length": 20,
        "IOU":        {"iou_min": 0.01},
        "NND":        {"max_distance": 30.0, "gap_closing": 0},
        "trackastra": {"remove_gaps": True},
    }

    # step 3
    config["step3_cropping"] = {"margin": 30}

    # step 4
    config["step4_registration"] = {
        "method":        reg_method,
        "preprocessing": None,
        "elastix":       {"loss": "MSE"},
    }

    # step 5
    config["step5_detection"] = {
        "method":              foci_detector,
        "threshold":           None,   # None = use per-method default
        "return_segmentation": False,
        "params":              {},
    }

    # step 6
    config["step6_tracking"] = {
        "method":           foci_tracker,
        "min_track_length": 3,
        "GNN":              {"max_distance": 5.0, "gap_closing": 2},
        "NGMA":             {"max_distance": 5.0, "gap_closing": 2},
        "trackastra":       {"use_segmentation": True},
    }

    # ── write yaml ────────────────────────────────────────────────────────────
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\nGenerated parameters.yml at {out}\n")
    print("── installed packages detected ──────────────────────────────")
    print(f"   cellpose-SAM  (cp-sam):   {'✓' if has_cp_sam     else '✗  pip install \"LiFT[cp-sam]\"'}")
    print(f"   cellpose-v3   (cp-v3):    {'✓' if has_cp_v3      else '✗  pip install \"LiFT[cp-v3]\"'}")
    print(f"   trackastra:               {'✓' if has_trackastra else '✗  pip install \"LiFT[trackastra]\"'}")
    print(f"   spotiflow:                {'✓' if has_spotiflow  else '✗  pip install \"LiFT[spotiflow]\"'}")
    print(f"   itk-elastix:              {'✓' if has_elastix    else '✗  pip install \"LiFT[elastix]\"'}")
    print("─────────────────────────────────────────────────────────────")
    print(f"\n── defaults written ─────────────────────────────────────────")
    print(f"   segmentation:  {seg_method or 'none — install cp-sam or cp-v3'}")
    print(f"   nuc tracking:  {nuc_tracker}")
    print(f"   registration:  {reg_method}")
    print(f"   foci detector: {foci_detector}")
    print(f"   foci tracking: {foci_tracker}")
    print("─────────────────────────────────────────────────────────────\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LiFT – Live Foci Tracking pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    # lift run <data_path>
    run_p = sub.add_parser("run", help="Run pipeline from parameters.yml")
    run_p.add_argument("data_path", help="Path to data folder")
    run_p.add_argument(
        "--steps", nargs="+", type=int, metavar="N",
        help="Steps to run, e.g. --steps 1 2 5"
    )

    # lift init <data_path>
    init_p = sub.add_parser("init", help="Generate parameters.yml from installed packages")
    init_p.add_argument("data_path", help="Path to write parameters.yml into")

    args = parser.parse_args()

    if args.command == "run":
        run(args.data_path, steps=args.steps)
    elif args.command == "init":
        init(args.data_path)
    else:
        # no subcommand — preserve old behaviour so `lift data/` still works
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("data_path")
        parser2.add_argument("--steps", nargs="+", type=int, metavar="N")
        args2 = parser2.parse_args()
        run(args2.data_path, steps=args2.steps)