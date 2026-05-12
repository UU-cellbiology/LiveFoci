"""LiFT – Live Foci Tracking.

Reproduces a full pipeline run using a parameters.yml file that was saved
by the GUI or notebook when each step was run. you can run this both from the
command line or incorporate it into your custom python file.

Usage:
    python LiFT.py data/
    python LiFT.py data/ --steps 1 2 5 6   # run specific steps only

Programmatic:
    import LiFT
    LiFT.run("data/")
    LiFT.run("data/", steps=[1, 2, 5, 6])
"""

import sys
from pathlib import Path
from pipeline.general_utils.params_utils import load_params


def run(data_path: str, steps: list = None):
    """Run pipeline steps defined in parameters.yml inside *data_path*.

    steps: list of ints (1-6) to execute; None runs all present sections.
    """
    data_path = Path(data_path)
    p = load_params(data_path)

    if not p:
        print(f"No parameters.yml found in {data_path}. Nothing to run.")
        return

    def _should_run(step_num):
        return steps is None or step_num in steps

    #### step 1: Nuclei Segmentation 
    if _should_run(1) and (s := p.get('step1_segmentation')):
        print("── Step 1: Nuclei Segmentation")
        from pipeline import nuc_segmentation
        
        seg          = s.get('segmentation', {})
        prepr        = s.get('preprocessing', {})
        method_name  = seg.get('method', 'cellpose_sam')
        preproc_name = prepr.get('method')
        min_area     = seg.get('min_area', 1000)
        method_params = {'flow_threshold':     seg.get('flow_threshold',      0.0), 'cellprob_threshold': seg.get('cellprob_threshold', -0.5), **seg.get(method_name, {})}
        
        method         = _make_seg_method(method_name, method_params)
        preproc_func   = _make_preproc(preproc_name)
        preproc_kwargs = prepr.get(preproc_name, {}) if preproc_name not in (None, 'None', '') else {}
        
        raw_paths = sorted(data_path.glob("*/*/Pos*/raw"))
        nuc_segmentation.segment_folderlist(raw_paths, method, preproc_func, min_area=min_area, **preproc_kwargs)


    ##### step 2: Nuclei Tracking 
    if _should_run(2) and (s := p.get('step2_tracking')):
        print("── Step 2: Nuclei Tracking")
        from pipeline import nuclei_trackers
        
        method        = s['method']
        min_length    = s.get('min_length', 20)
        method_params = s.get(method, {})
        seg_paths = sorted(data_path.glob("*/*/Pos*/results/result_cell_seg"))
        nuclei_trackers.run_tracker(seg_paths, method=method, min_length=min_length, **method_params)


    ##### step 3: Cell Cropping
    if _should_run(3) and (s := p.get('step3_cropping')):
        print("── Step 3: Cell Cropping")
        from pipeline import cut_out_cells
        
        for pp in sorted(data_path.glob("*/*/Pos*")):
            try:
                cut_out_cells.cut_from_ctc(pp, margin=s.get('margin', 30))
            except ValueError:
                pass  # output folder already exists


    ##### step 4: Registration 
    if _should_run(4) and (s := p.get('step4_registration')):
        print("── Step 4: Registration")
        from pipeline import registration
        
        method       = s['method']
        preproc_func = _make_preproc(s.get('preprocessing'))
        kwargs       = s.get('elastix', {}) if method == 'elastix' else {}
        followed = sorted(data_path.glob("*/*/Pos*/results/followed/*/I_*.tif"))
        registration.run_registration(followed, method=method, preprocess_function=preproc_func, **kwargs)


    ##### step 5: Foci Detection 
    if _should_run(5) and (s := p.get('step5_detection')):
        print("── Step 5: Foci Detection")
        from pipeline import foci_detection
        
        s = dict(s)
        method_params    = s.pop('params', {})
        method           = s.pop('method')
        threshold        = s.pop('threshold')
        return_seg       = s.pop('return_segmentation', False)
        
        detector, _      = foci_detection.create_detector(method, params=method_params, threshold=threshold)
        
        registered = sorted(data_path.glob("*/*/Pos*/results/registered/*/I_*.tif"))
        foci_detection.run_detection(registered, detector, threshold, return_segmentation=return_seg)


    ##### step 6: Foci Tracking 
    if _should_run(6) and (s := p.get('step6_tracking')):
        print("── Step 6: Foci Tracking")
        from pipeline import foci_trackers
        
        method           = s['method']
        min_track_length = s.get('min_track_length', 3)
        method_params    = s.get(method, {})
        
        registered = sorted(data_path.glob("*/*/Pos*/results/registered/*/I_*.tif"))
        
        foci_trackers.run_foci_tracker(registered, method=method, min_track_length=min_track_length, **method_params)

    print("Done.")


# ── helper: reconstruct segmentation method object from YAML ─────────────────

def _make_seg_method(name, params):
    from pipeline import nuc_segmentation
    if name == 'cellpose_sam':
        return nuc_segmentation.CP_SAM(
            flow_threshold     = params.get('flow_threshold',      0.0),
            cellprob_threshold = params.get('cellprob_threshold', -0.5),
            scale_factor       = params.get('scale_factor',          1),
        )
    if name == 'cellpose_v3':
        return nuc_segmentation.CP_V3(
            flow_threshold     = params.get('flow_threshold',      0.0),
            cellprob_threshold = params.get('cellprob_threshold', -0.5),
            diameter           = params.get('diameter',           140),
        )
    raise ValueError(f"Unknown segmentation method: {name!r}")


def _make_preproc(name):
    if name in (None, 'None', ''):
        return None
    from pipeline import registration as reg_mod, nuc_segmentation as seg_mod
    mapping = {
        'wavelet_denoise':  reg_mod.wavelet_denoise,
        'threshold':        reg_mod.threshold,
        'DOG_filter':       reg_mod.DOG_filter,
        'wavelet_filtering': seg_mod.wavelet_filtering,
        'contrast_adjuster': seg_mod.contrast_adjuster,
    }
    func = mapping.get(name)
    if func is None:
        raise ValueError(f"Unknown preprocessing function: {name!r}")
    return func


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run LiFT pipeline from parameters.yml")
    parser.add_argument('data_path', help="Path to the data folder containing parameters.yml")
    parser.add_argument('--steps', nargs='+', type=int, metavar='N',
                        help="Step numbers to run (default: all). E.g. --steps 5 6")
    args = parser.parse_args()
    
    run(args.data_path, steps=args.steps)
