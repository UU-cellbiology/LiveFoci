from pathlib import Path
import yaml


def yaml_path(data_path):
    return Path(data_path) / "parameters.yml"


def load_params(data_path):
    p = yaml_path(data_path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_section(data_path, section, params):
    """Merge params into parameters.yml under the given section key."""
    doc = load_params(data_path)
    doc[section] = params
    with open(yaml_path(data_path), 'w') as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


# ── per-step save functions ───────────────────────────────────────────────────

def save_segmentation_params(data_path, seg_method, flow_threshold, cellprob_threshold,
                              scale_factor, diameter, min_area,
                              preproc, scales, w_factor, start_scale, sigma, c_factor):
    save_section(data_path, 'step1_segmentation', {
        'segmentation': {
            'method':             seg_method,
            'flow_threshold':     flow_threshold,
            'cellprob_threshold': cellprob_threshold,
            'min_area':           min_area,
            'cellpose_sam':       {'scale_factor': scale_factor},
            'cellpose_v3':        {'diameter': diameter},
        },
        'preprocessing': {
            'method':            preproc,
            'wavelet_filtering': {'scales': scales, 'w_factor': w_factor, 'start_scale': start_scale},
            'contrast_adjuster': {'sigma': sigma, 'c_factor': c_factor},
        },
    })


def save_tracking_params(data_path, method, min_length, max_distance, gap_closing,
                         iou_min, remove_gaps):
    save_section(data_path, 'step2_tracking', {
        'method':     method,
        'min_length': min_length,
        'NND':        {'max_distance': max_distance, 'gap_closing': gap_closing},
        'IOU':        {'iou_min': iou_min},
        'trackastra': {'remove_gaps': remove_gaps},
    })


def save_cropping_params(data_path, margin):
    save_section(data_path, 'step3_cropping', {'margin': margin})


def save_registration_params(data_path, method, preprocessing, loss):
    save_section(data_path, 'step4_registration', {
        'method':        method,
        'preprocessing': preprocessing,
        'elastix':       {'loss': loss},
    })


def save_detection_params(data_path, method, threshold, return_segmentation, params):
    save_section(data_path, 'step5_detection', {
        'method':             method,
        'threshold':          threshold,
        'return_segmentation': return_segmentation,
        'params':             params,
    })


def save_foci_tracking_params(data_path, method, min_track_length, max_distance,
                               gap_closing, use_segmentation):
    save_section(data_path, 'step6_tracking', {
        'method':           method,
        'min_track_length': min_track_length,
        'GNN':              {'max_distance': max_distance, 'gap_closing': gap_closing},
        'NGMA':             {'max_distance': max_distance, 'gap_closing': gap_closing},
        'trackastra':       {'use_segmentation': use_segmentation},
    })
