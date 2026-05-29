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

def init(data_path: str = None):
    """
    Generate a parameters.yml in data_path based on what is installed.
    Defaults to current working directory if no path given.
    """
    from lift._helpers import _probe_version
    from lift.general_utils.params_utils import yaml_path
    from pathlib import Path
    import yaml

    data_path = Path(data_path).resolve() if data_path else Path.cwd()
    data_path.mkdir(parents=True, exist_ok=True)

    out = yaml_path(data_path)
    if out.exists():
        print(f"parameters.yml already exists at {out} — skipping.")
        return

    # ── probe installed packages ──────────────────────────────────────────────
    cellpose_v      = _probe_version("cellpose")
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

    config["step2_tracking"] = {
        "method":     nuc_tracker,
        "min_length": 20,
        "IOU":        {"iou_min": 0.01},
        "NND":        {"max_distance": 30.0, "gap_closing": 0},
        "trackastra": {"remove_gaps": True},
    }

    config["step3_cropping"] = {"margin": 30}

    config["step4_registration"] = {
        "method":        reg_method,
        "preprocessing": None,
        "elastix":       {"loss": "MSE"},
    }

    config["step5_detection"] = {
        "method":              foci_detector,
        "threshold":           None,
        "return_segmentation": False,
        "params":              {},
    }

    config["step6_tracking"] = {
        "method":           foci_tracker,
        "min_track_length": 3,
        "GNN":              {"max_distance": 5.0, "gap_closing": 2},
        "NGMA":             {"max_distance": 5.0, "gap_closing": 2},
        "trackastra":       {"use_segmentation": True},
    }

    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    cp_sam_str     = '✓' if has_cp_sam     else '✗  pip install "LiFT[cp-sam]"'
    cp_v3_str      = '✓' if has_cp_v3      else '✗  pip install "LiFT[cp-v3]"'
    trackastra_str = '✓' if has_trackastra else '✗  pip install "LiFT[trackastra]"'
    spotiflow_str  = '✓' if has_spotiflow  else '✗  pip install "LiFT[spotiflow]"'
    elastix_str    = '✓' if has_elastix    else '✗  pip install "LiFT[elastix]"'

    print(f"\nGenerated parameters.yml at {out}\n")
    print("── installed packages detected ──────────────────────────────")
    print(f"   cellpose-SAM  (cp-sam):   {cp_sam_str}")
    print(f"   cellpose-v3   (cp-v3):    {cp_v3_str}")
    print(f"   trackastra:               {trackastra_str}")
    print(f"   spotiflow:                {spotiflow_str}")
    print(f"   itk-elastix:              {elastix_str}")
    print("─────────────────────────────────────────────────────────────")
    print(f"\n── defaults written ─────────────────────────────────────────")
    print(f"   segmentation:  {seg_method or 'none — install cp-sam or cp-v3'}")
    print(f"   nuc tracking:  {nuc_tracker}")
    print(f"   registration:  {reg_method}")
    print(f"   foci detector: {foci_detector}")
    print(f"   foci tracking: {foci_tracker}")
    print("─────────────────────────────────────────────────────────────\n")

def info():
    """Print installed LiFT-relevant packages and their versions."""
    import importlib

    packages = [
        ("cellpose",   "cp-sam / cp-v3", "cellpose>=4.0 or cellpose>=3.0,<4.0"),
        ("trackastra", "trackastra",      "trackastra>=0.2"),
        ("spotiflow",  "spotiflow",       "spotiflow>=0.4"),
        ("itk",        "elastix",         "itk-elastix>=5.3"),
        ("torch",      "cp-sam/cp-v3",    "installed by cellpose"),
    ]

    print("\n── LiFT environment ─────────────────────────────────────────")
    for package, extra, note in packages:
        try:
            mod = importlib.import_module(package)
            v   = getattr(mod, "__version__", "unknown version")
            print(f"   ✓  {package:<16} {v:<12}  [{extra}]")
        except ImportError:
            print(f"   ✗  {package:<16} not installed   pip install \"LiFT[{extra}]\"")
    print("─────────────────────────────────────────────────────────────\n")

def config_cmd(data_path: str = None, sets: list = None):
    """
    Interactively edit parameters.yml, or set values directly with key.path=value.
    Defaults to current working directory if no path given.
    """
    from lift.general_utils.params_utils import load_params, yaml_path
    from pathlib import Path
    import yaml

    sets = sets or []

    # if data_path looks like a key=value pair, treat it as a set
    if data_path and "=" in data_path:
        sets = [data_path] + sets
        data_path = None

    data_path = Path(data_path).resolve() if data_path else Path.cwd()
    out       = yaml_path(data_path)

    if not out.exists():
        print(f"No parameters.yml found at {out}. Run 'lift init' first.")
        return

    params = load_params(data_path)

    # ── non-interactive mode: sets provided directly ──────────────────────────
    if sets:
        for item in sets:
            if "=" not in item:
                print(f"  ✗ skipping {item!r} — expected format: key.path=value")
                continue
            key_path, raw_value = item.split("=", 1)
            value = _parse_value(raw_value)
            keys  = key_path.strip().split(".")
            d     = params
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
            print(f"  ✓ {key_path} = {value!r}")

        with open(out, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        print(f"\nSaved to {out}")
        return

    # ── interactive mode ──────────────────────────────────────────────────────
    print(f"\n── Interactive config editor ─────────────────────────────────")
    print(f"   {out}")
    print(f"   Press Enter to keep current value. Type new value to change.")
    print(f"─────────────────────────────────────────────────────────────\n")

    _edit_section(params, "step1_segmentation", [
        ("segmentation.method",             "Segmentation method",          ["cellpose_sam", "cellpose_v3"]),
        ("segmentation.flow_threshold",     "Flow threshold",               None),
        ("segmentation.cellprob_threshold", "Cell probability threshold",   None),
        ("segmentation.min_area",           "Min nucleus area (px)",        None),
        ("segmentation.cellpose_sam.scale_factor", "CP-SAM scale factor",   None),
        ("segmentation.cellpose_v3.diameter",      "CP-v3 diameter (px)",   None),
        ("preprocessing.method",            "Preprocessing method",         ["null", "wavelet_filtering", "contrast_adjuster"]),
        ("preprocessing.wavelet_filtering.scales",      "Wavelet scales",   None),
        ("preprocessing.wavelet_filtering.w_factor",    "Wavelet factor",   None),
        ("preprocessing.wavelet_filtering.start_scale", "Wavelet start scale", None),
        ("preprocessing.contrast_adjuster.sigma",    "Contrast sigma",      None),
        ("preprocessing.contrast_adjuster.c_factor", "Contrast factor",     None),
    ])

    _edit_section(params, "step2_tracking", [
        ("method",               "Tracking method",              ["IOU", "NND", "trackastra"]),
        ("min_length",           "Min track length (frames)",    None),
        ("IOU.iou_min",          "IOU min overlap",              None),
        ("NND.max_distance",     "NND max distance (px)",        None),
        ("NND.gap_closing",      "NND gap closing (frames)",     None),
        ("trackastra.remove_gaps", "Trackastra remove gaps",     ["true", "false"]),
    ])

    _edit_section(params, "step3_cropping", [
        ("margin", "Crop margin (px)", None),
    ])

    _edit_section(params, "step4_registration", [
        ("method",        "Registration method",      ["stackreg", "elastix"]),
        ("preprocessing", "Preprocessing function",   ["null", "wavelet_denoise", "DOG_filter", "threshold"]),
        ("elastix.loss",  "Elastix loss function",    ["MSE", "MI", "NCC"]),
    ])

    _edit_section(params, "step5_detection", [
        ("method",              "Detection method",     ["Wavelets", "LOG", "Hessian", "TopHat", "HDome", "HDome-smal", "MPHD", "Spotiflow"]),
        ("threshold",           "Detection threshold",  None),
        ("return_segmentation", "Return segmentation",  ["true", "false"]),
        ("params.K",            "Wavelets K",           None),
        ("params.factor",       "Wavelets factor",      None),
        ("params.start_scale",  "Wavelets start scale", None),
        ("params.sigma",        "Sigma",                None),
        ("params.radius",       "TopHat radius",        None),
        ("params.h",            "HDome h",              None),
        ("params.h_init",       "MPHD h_init",          None),
        ("params.R",            "MPHD R",               None),
        ("params.model_name",   "Spotiflow model",      ["general"]),
    ])

    _edit_section(params, "step6_tracking", [
        ("method",                    "Foci tracking method",         ["GNN", "NGMA", "trackastra"]),
        ("min_track_length",          "Min track length",             None),
        ("GNN.max_distance",          "GNN max distance (px)",        None),
        ("GNN.gap_closing",           "GNN gap closing (frames)",     None),
        ("NGMA.max_distance",         "NGMA max distance (px)",       None),
        ("NGMA.gap_closing",          "NGMA gap closing (frames)",    None),
        ("trackastra.use_segmentation", "Trackastra use segmentation", ["true", "false"]),
    ])

    with open(out, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    print(f"\n── Saved to {out} ───────────────────────────────────────────\n")


def _edit_section(params, section_key, fields):
    """Walk through fields in a section and prompt user to edit each."""

    if section_key not in params:
        print(f"\n  (skipping {section_key} — not in config)\n")
        return

    section = params[section_key]
    label   = section_key.replace("step", "Step ").replace("_", " ").title()

    print(f"── {label} {'─' * (50 - len(label))}")

    for key_path, description, options in fields:
        # get current value
        keys    = key_path.split(".")
        current = section
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                current = None
                break

        # build prompt
        if options:
            opts_str = " / ".join(options)
            prompt   = f"  {description} [{opts_str}] (current: {current}): "
        else:
            prompt   = f"  {description} (current: {current}): "

        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — saving what was changed so far.")
            return

        if raw == "":
            continue  # keep current value

        value = _parse_value(raw)

        # set value back into section
        d = section
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        print(f"  ✓ set to {value!r}")

    print()

def _parse_value(raw: str):
    """Parse a string value into the most appropriate Python type."""
    if raw.lower() == "null" or raw.lower() == "none":
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw

def main():
    import argparse
    import sys

    known_commands = {"run", "init", "config", "info"}
    if len(sys.argv) < 2 or sys.argv[1] not in known_commands:
        parser = argparse.ArgumentParser(description="LiFT – Live Foci Tracking pipeline")
        parser.add_argument("data_path")
        parser.add_argument("--steps", nargs="+", type=int, metavar="N")
        args = parser.parse_args()
        run(args.data_path, steps=args.steps)
        return

    parser = argparse.ArgumentParser(description="LiFT – Live Foci Tracking pipeline")
    sub    = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run pipeline from parameters.yml")
    run_p.add_argument("data_path")
    run_p.add_argument("--steps", nargs="+", type=int, metavar="N")

    init_p = sub.add_parser("init", help="Generate parameters.yml from installed packages")
    init_p.add_argument(
        "data_path", nargs="?", default=None,
        help="Path to write parameters.yml into (default: current directory)"
    )

    cfg_p = sub.add_parser("config", help="Set or view values in parameters.yml")
    cfg_p.add_argument(
        "data_path", nargs="?", default=None,
        help="Path to data folder containing parameters.yml (default: current directory)"
    )
    cfg_p.add_argument(
        "sets", nargs="*", metavar="key.path=value",
        help="One or more key.path=value pairs to set. Omit to print current config."
    )

    info_p = sub.add_parser("info", help="Show information about installed LiFT components")

    args = parser.parse_args()

    if args.command == "run":
        run(args.data_path, steps=args.steps)
    elif args.command == "init":
        init(args.data_path)
    elif args.command == "config":
        config_cmd(args.data_path, args.sets)
    elif args.command == "info":
        info()