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

        raw_paths = sorted(data_path.glob("*/*/Pos*/raw"))
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

        seg_paths = sorted(data_path.glob("*/*/Pos*/results/result_cell_seg"))
        nuclei_trackers.run_tracker(
            seg_paths, method=method, min_length=min_length, **method_params
        )

    # ── step 3: cell cropping ────────────────────────────────────────────────
    if _should_run(3) and (s := p.get('step3_cropping')):
        print("── Step 3: Cell Cropping")
        from lift import cut_out_cells

        for pp in sorted(data_path.glob("*/*/Pos*")):
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

        followed = sorted(data_path.glob("*/*/Pos*/results/followed/*/I_*.tif"))
        registration.run_registration(
            followed, method=method, preprocess_function=preproc_func, **kwargs
        )

    # ── step 5: foci detection ───────────────────────────────────────────────
    if _should_run(5) and (s := p.get('step5_detection')):
        print("── Step 5: Foci Detection")
        from lift import foci_detection

        s = dict(s)
        method_params    = s.pop('params', {})
        method           = s.pop('method', None)
        threshold        = s.pop('threshold', None)
        return_seg       = s.pop('return_segmentation', False)

        detector, threshold = foci_detection.create_detector(
            method, params=method_params, threshold=threshold
        )

        registered = sorted(data_path.glob("*/*/Pos*/results/registered/*/I_*.tif"))
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

        registered = sorted(data_path.glob("*/*/Pos*/results/registered/*/I_*.tif"))
        foci_trackers.run_foci_tracker(
            registered, method=method, min_track_length=min_track_length, **method_params
        )

    print("Done.")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run LiFT pipeline from parameters.yml"
    )
    parser.add_argument(
        'data_path',
        help="Path to the data folder containing parameters.yml"
    )
    parser.add_argument(
        '--steps', nargs='+', type=int, metavar='N',
        help="Step numbers to run (default: all). E.g. --steps 5 6"
    )
    args = parser.parse_args()
    run(args.data_path, steps=args.steps)