from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import uuid
import numpy as np
from pathlib import Path
import skimage.io
import skimage.color
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lift import nuc_segmentation, nuclei_trackers, cut_out_cells, registration, foci_detection, foci_trackers
from lift.general_utils import load_sequence, plt_figures
from lift.general_utils.params_utils import (
    save_segmentation_params, save_tracking_params, save_cropping_params,
    save_registration_params, save_detection_params, save_foci_tracking_params,
)
from PIL import Image

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data")

# In-memory store for background task progress
tasks: dict = {}
_preview_lock = threading.Lock()


##############################################################################
# Image helpers
##############################################################################

def array_to_b64png(arr, vmax=None, max_size=350):
    """Convert a 2-D grayscale or 3-D RGB (float 0-1) numpy array to a
    base64-encoded PNG string, thumbnail-scaled for web display."""
    arr = np.array(arr, dtype=float)

    if arr.ndim == 2:
        if vmax is not None:
            arr = np.clip(arr / vmax, 0, 1)
        else:
            lo, hi = arr.min(), arr.max()
            arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
    else:  # RGB float
        arr = np.clip(arr, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')

    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

##############################################################################
# Page routes
##############################################################################

@app.route('/logo.png')
def serve_logo():
    return send_from_directory(PROJECT_ROOT / 'templates', 'logo.png')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/step1')
def step1():
    return render_template('step1_segmentation.html', default_data_path=DEFAULT_DATA_PATH)


##############################################################################
# API – discover raw-image folders
##############################################################################

@app.route('/api/discover_paths', methods=['POST'])
def discover_paths():
    data = request.get_json()
    data_path = data.get('data_path', DEFAULT_DATA_PATH)
    pattern   = data.get('pattern', '*/*/Pos*/raw')
    try:
        raw_paths = sorted(Path(data_path).glob(pattern))
        return jsonify({
            'success': True,
            'paths': [str(p) for p in raw_paths],
            'count': len(raw_paths),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


##############################################################################
# API – segmentation preview (synchronous, locked)
##############################################################################

@app.route('/api/preview_segmentation', methods=['POST'])
def preview_segmentation():
    if not _preview_lock.acquire(blocking=False):
        return jsonify({'success': False,
                        'error': 'A preview is already running – please wait.'})
    try:
        data = request.get_json()
        raw_paths = data.get('raw_paths', [])
        if not raw_paths:
            return jsonify({'success': False,
                            'error': 'No folders found. Click "Discover" first.'})

        # parameters
        preproc_name = data.get('preprocessing_method', 'None')
        sigma        = float(data.get('sigma',       3.0))
        c_factor     = float(data.get('c_factor',    6.0))
        scales       = int(data.get('scales',        5))
        w_factor     = float(data.get('w_factor',    1.5))
        start_scale  = int(data.get('start_scale',   2))
        max_int      = float(data.get('max_int',     60))

        seg_method   = data.get('seg_method',  'CP_SAM')
        flow_T       = float(data.get('flow_T',       0.0))
        prob_T       = float(data.get('prob_T',      -0.5))
        scale_factor = float(data.get('scale_factor', 0.5))
        diameter     = float(data.get('diameter',    140))

        # pick a random folder and sample 5 frames
        selected_path = Path(raw_paths[np.random.randint(len(raw_paths))])
        raw_stack = load_sequence.load(selected_path)
        indices   = np.linspace(0, len(raw_stack) - 1, 5, dtype=int)
        test_data = [raw_stack[i].copy() for i in indices]

        # preprocessing
        preproc_func = {
            'contrast_adjuster': nuc_segmentation.contrast_adjuster,
            'wavelet_filtering': nuc_segmentation.wavelet_filtering,
        }.get(preproc_name)

        processed = nuc_segmentation.preprocessing(
            test_data, preproc_func,
            scales=scales, w_factor=w_factor, start_scale=start_scale,
            sigma=sigma, c_factor=c_factor,
        )

        # segmentation
        if seg_method == 'CP_SAM':
            method = nuc_segmentation.CP_SAM(flow_threshold=flow_T, cellprob_threshold=prob_T, scale_factor=scale_factor)
        else:
            method = nuc_segmentation.CP_V3(flow_threshold=flow_T, cellprob_threshold=prob_T, diameter=diameter)

        preds = nuc_segmentation.segment_imglist(processed, method)

        # encode result images
        result_images = []
        for ii in range(len(test_data)):
            overlay = skimage.color.label2rgb(preds[ii], test_data[ii], bg_label=0)
            result_images.append({
                'raw':       array_to_b64png(test_data[ii], vmax=max_int),
                'processed': array_to_b64png(processed[ii]),
                'segmented': array_to_b64png(overlay),
                'frame':     int(indices[ii]),
                'n_nuclei':  int(preds[ii].max()),
            })

        return jsonify({
            'success': True,
            'images': result_images,
            'selected_path': str(selected_path),
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})
    finally:
        _preview_lock.release()


##############################################################################
# API – run full segmentation in background
##############################################################################

@app.route('/api/run_segmentation', methods=['POST'])
def run_segmentation():
    data = request.get_json()
    raw_paths = data.get('raw_paths', [])
    if not raw_paths:
        return jsonify({'success': False, 'error': 'No folders to process.'})

    # Check if a job is already running
    for t in tasks.values():
        if t.get('status') == 'running':
            return jsonify({'success': False,
                            'error': 'A segmentation job is already running.'})

    # parameters
    data_path    = data.get('data_path', DEFAULT_DATA_PATH)
    preproc_name = data.get('preprocessing_method', 'None')
    sigma        = float(data.get('sigma',       3.0))
    c_factor     = float(data.get('c_factor',    6.0))
    scales       = int(data.get('scales',        5))
    w_factor     = float(data.get('w_factor',    1.5))
    start_scale  = int(data.get('start_scale',   2))

    seg_method   = data.get('seg_method',  'CP_SAM')
    flow_T       = float(data.get('flow_T',       0.0))
    prob_T       = float(data.get('prob_T',      -0.5))
    scale_factor = float(data.get('scale_factor', 0.5))
    diameter     = float(data.get('diameter',    140))
    min_area     = int(data.get('min_area',     1000))

    save_segmentation_params(data_path, seg_method='cellpose_sam' if seg_method == 'CP_SAM' else 'cellpose_v3',
        flow_threshold=flow_T, cellprob_threshold=prob_T, scale_factor=scale_factor,
        diameter=diameter, min_area=min_area, preproc=preproc_name, scales=scales,
        w_factor=w_factor, start_scale=start_scale, sigma=sigma, c_factor=c_factor)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running', 'progress': 0,
        'total': len(raw_paths), 'log': [],
    }

    preproc_func = {
        'contrast_adjuster': nuc_segmentation.contrast_adjuster,
        'wavelet_filtering': nuc_segmentation.wavelet_filtering,
    }.get(preproc_name)

    def background_job():
        try:
            # Build the segmentation model once for the whole run
            if seg_method == 'CP_SAM':
                method = nuc_segmentation.CP_SAM(flow_threshold=flow_T, cellprob_threshold=prob_T, scale_factor=scale_factor)
            else:
                method = nuc_segmentation.CP_V3(flow_threshold=flow_T, cellprob_threshold=prob_T, diameter=diameter)

            for i, folder_str in enumerate(raw_paths):
                folder = Path(folder_str)
                tasks[task_id]['progress'] = i
                tasks[task_id]['log'].append(
                    f'[{i+1}/{len(raw_paths)}] {folder.parent.parent.name} / {folder.parent.name}')

                nuc_segmentation.segment_folderlist([folder], method, preproc_func, min_area=min_area,
                    scales=scales, w_factor=w_factor, start_scale=start_scale, sigma=sigma, c_factor=c_factor)

                tasks[task_id]['log'].append('  saved')

            tasks[task_id]['progress'] = len(raw_paths)
            tasks[task_id]['status']   = 'done'
            tasks[task_id]['log'].append('All done!')

        except Exception as e:
            import traceback
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error']  = str(e)
            tasks[task_id]['log'].append(f'ERROR: {e}')
            tasks[task_id]['log'].append(traceback.format_exc())

    threading.Thread(target=background_job, daemon=True).start()
    return jsonify({'success': True, 'task_id': task_id, 'yaml_saved': True})


@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    if task_id not in tasks:
        return jsonify({'success': False, 'error': 'Task not found'})
    return jsonify({'success': True, **tasks[task_id]})


##############################################################################
# Step 2 – Nuclei Tracking
##############################################################################

@app.route('/step2')
def step2():
    return render_template('step2_tracking.html', default_data_path=DEFAULT_DATA_PATH)


@app.route('/api/preview_tracking', methods=['POST'])
def preview_tracking():
    data = request.get_json()
    seg_paths = data.get('seg_paths', [])
    if not seg_paths:
        return jsonify({'success': False,
                        'error': 'No segmentation folders found. Click "Discover" first.'})

    # parameters
    method      = data.get('tracking_method', 'NND')
    min_length  = int(data.get('min_length',   36))
    iou_min     = float(data.get('iou_min',    0.01))
    max_dist    = float(data.get('max_distance', 30.0))
    gap_closing = int(data.get('gap_closing',   0))
    remove_gaps = bool(data.get('remove_gaps',  True))
    max_int     = float(data.get('max_int',     100))

    try:
        # pick one random position and run tracking 
        selected_path = Path(seg_paths[np.random.randint(len(seg_paths))])

        nuclei_trackers.run_tracker([selected_path], method=method, min_length=min_length,
            iou_min=iou_min, max_distance=max_dist, gap_closing=gap_closing, remove_gaps=remove_gaps)

        # check tracking produced output 
        tracking_path = selected_path.parent / 'cell_tracking'
        tracked_files = sorted(tracking_path.glob('t*.tif'))
        if not tracked_files:
            return jsonify({'success': False,
                            'error': 'Tracking produced no output files.'})

        # count tracks from res_track.txt 
        res_track = tracking_path / 'res_track.txt'
        n_tracks  = 0
        if res_track.exists():
            try:
                track_data = np.loadtxt(res_track, dtype=int)
                if track_data.ndim == 1:
                    track_data = track_data[np.newaxis, :]
                n_tracks = len(track_data)
            except Exception:
                pass

        # generate movie then embed as base64
        # animate_from_masks expects the 'results' folder as base_path and
        # saves the movie to  base_path / 'cell_tracking' / 'tracking.mp4'
        results_path = selected_path.parent
        plt_figures.animate_from_masks(results_path, dpi=120, fps=7, vmax=max_int)

        movie_path = tracking_path / 'tracking.mp4'
        if not movie_path.exists():
            return jsonify({'success': False,
                            'error': 'Movie file was not created (is ffmpeg installed?).'})

        with open(movie_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            'success':       True,
            'movie_b64':     video_b64,
            'selected_path': str(selected_path),
            'n_tracks':      n_tracks,
            'n_frames':      len(tracked_files),
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


@app.route('/api/run_tracking', methods=['POST'])
def run_tracking():
    data = request.get_json()
    seg_paths = data.get('seg_paths', [])
    if not seg_paths:
        return jsonify({'success': False, 'error': 'No folders to process.'})

    for t in tasks.values():
        if t.get('status') == 'running':
            return jsonify({'success': False,
                            'error': 'A job is already running.'})

    data_path   = data.get('data_path', DEFAULT_DATA_PATH)
    method      = data.get('tracking_method', 'NND')
    min_length  = int(data.get('min_length',    36))
    iou_min     = float(data.get('iou_min',     0.01))
    max_dist    = float(data.get('max_distance', 30.0))
    gap_closing = int(data.get('gap_closing',    0))
    remove_gaps = bool(data.get('remove_gaps',   True))

    save_tracking_params(data_path, method=method, min_length=min_length, max_distance=max_dist,
        gap_closing=gap_closing, iou_min=iou_min, remove_gaps=remove_gaps )

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running', 'progress': 0,
        'total': len(seg_paths), 'log': [],
    }

    def background_job():
        try:
            for i, seg_str in enumerate(seg_paths):
                seg_path = Path(seg_str)
                tasks[task_id]['progress'] = i
                tasks[task_id]['log'].append(f'[{i+1}/{len(seg_paths)}] {seg_path.parent.parent.name} / {seg_path.parent.name}')

                nuclei_trackers.run_tracker([seg_path], method=method, min_length=min_length,
                    iou_min=iou_min, max_distance=max_dist, gap_closing=gap_closing, remove_gaps=remove_gaps)

                tasks[task_id]['log'].append('  saved')

            tasks[task_id]['progress'] = len(seg_paths)
            tasks[task_id]['status']   = 'done'
            tasks[task_id]['log'].append('All done!')

        except Exception as e:
            import traceback
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error']  = str(e)
            tasks[task_id]['log'].append(f'ERROR: {e}')
            tasks[task_id]['log'].append(traceback.format_exc())

    threading.Thread(target=background_job, daemon=True).start()
    return jsonify({'success': True, 'task_id': task_id, 'yaml_saved': True})


##############################################################################
# Step 3 – Cell Cropping
##############################################################################

@app.route('/step3')
def step3():
    return render_template('step3_cropping.html', default_data_path=DEFAULT_DATA_PATH)


@app.route('/api/preview_cropping', methods=['POST'])
def preview_cropping():
    data      = request.get_json()
    pos_paths = data.get('pos_paths', [])
    margin    = int(data.get('margin', 30))

    if not pos_paths:
        return jsonify({'success': False,
                        'error': 'No position folders found. Click "Discover" first.'})
    try:
        # prefer a position that has not been cropped yet
        np.random.shuffle(pos_paths)
        selected_path = None
        already_done  = False
        for p in pos_paths:
            followed = Path(p) / 'results' / 'followed'
            if not followed.exists():
                selected_path = Path(p)
                break
        if selected_path is None:          # all done → show one that exists
            selected_path = Path(pos_paths[0])
            already_done  = True

        followed_path = selected_path / 'results' / 'followed'

        # run cropping (only if not already done) 
        if not already_done:
            cut_out_cells.cut_from_ctc(selected_path, margin=margin)

        # collect cell TIF files and build thumbnail list 
        tif_files = sorted(followed_path.glob('*/I_*.tif'))
        if not tif_files:
            return jsonify({'success': False,
                            'error': 'No cell crops found in the followed folder.'})

        thumbnails = []
        for tif_file in tif_files[:12]:          # show at most 12 cells
            stack = skimage.io.imread(str(tif_file))
            if stack.ndim == 3:
                mid = stack[len(stack) // 2]
                n_frames = len(stack)
            else:
                mid = stack
                n_frames = 1
            thumbnails.append({
                'image':    array_to_b64png(mid),
                'name':     tif_file.parent.name,
                'n_frames': n_frames,
                'size':     f'{mid.shape[1]}×{mid.shape[0]} px',
            })

        return jsonify({
            'success':       True,
            'thumbnails':    thumbnails,
            'selected_path': str(selected_path),
            'n_cells':       len(tif_files),
            'already_done':  already_done,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


@app.route('/api/run_cropping', methods=['POST'])
def run_cropping():
    data      = request.get_json()
    pos_paths = data.get('pos_paths', [])
    margin    = int(data.get('margin', 30))

    if not pos_paths:
        return jsonify({'success': False, 'error': 'No position folders to process.'})

    for t in tasks.values():
        if t.get('status') == 'running':
            return jsonify({'success': False, 'error': 'A job is already running.'})

    data_path = data.get('data_path', DEFAULT_DATA_PATH)
    save_cropping_params(data_path, margin=margin)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running', 'progress': 0,
        'total': len(pos_paths), 'log': [],
    }

    def background_job():
        try:
            for i, pos_str in enumerate(pos_paths):
                pos_path = Path(pos_str)
                tasks[task_id]['progress'] = i
                label = f'[{i+1}/{len(pos_paths)}] {pos_path.parent.name} / {pos_path.name}'
                tasks[task_id]['log'].append(label)

                try:
                    cut_out_cells.cut_from_ctc(pos_path, margin=margin)
                    tasks[task_id]['log'].append('  saved')
                except ValueError:
                    tasks[task_id]['log'].append('  skipped — output already exists')

            tasks[task_id]['progress'] = len(pos_paths)
            tasks[task_id]['status']   = 'done'
            tasks[task_id]['log'].append('All done!')

        except Exception as e:
            import traceback
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error']  = str(e)
            tasks[task_id]['log'].append(f'ERROR: {e}')
            tasks[task_id]['log'].append(traceback.format_exc())

    threading.Thread(target=background_job, daemon=True).start()
    return jsonify({'success': True, 'task_id': task_id, 'yaml_saved': True})


##############################################################################
# Step 4 – Registration
##############################################################################

@app.route('/step4')
def step4():
    return render_template('step4_registration.html', default_data_path=DEFAULT_DATA_PATH)


@app.route('/api/preview_registration', methods=['POST'])
def preview_registration():
    data        = request.get_json()
    tif_paths   = data.get('followed_tifs', [])
    method      = data.get('method', 'stackreg')
    preproc_name = data.get('preproc', 'None')
    loss        = data.get('loss', 'MSE')

    if not tif_paths:
        return jsonify({'success': False,
                        'error': 'No cell stacks found. Click "Discover" first.'})

    _PREPROC = {
        'wavelet_denoise': registration.wavelet_denoise,
        'threshold':       registration.threshold,
        'DOG_filter':      registration.DOG_filter,
    }
    preproc_func = _PREPROC.get(preproc_name)   # None when 'None'

    try:
        tif_path = Path(tif_paths[np.random.randint(len(tif_paths))])
        stack    = skimage.io.imread(str(tif_path))      # (T, H, W)

        kwargs = {'loss': loss} if method == 'elastix' else {}
        registered = registration.register_array(stack, method, preproc_func, **kwargs)

        n = len(stack)
        indices = list(range(min(6, n)))

        return jsonify({
            'success':       True,
            'input':        [array_to_b64png(stack[i])      for i in indices],
            'registered':         [array_to_b64png(registered[i]) for i in indices],
            'frames':        indices,
            'selected_path': str(tif_path),
            'n_frames':      n,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


@app.route('/api/run_registration', methods=['POST'])
def run_registration():
    data        = request.get_json()
    tif_paths   = data.get('followed_tifs', [])
    method      = data.get('method', 'stackreg')
    preproc_name = data.get('preproc', 'None')
    loss        = data.get('loss', 'MSE')

    if not tif_paths:
        return jsonify({'success': False, 'error': 'No cell stacks to process.'})

    for t in tasks.values():
        if t.get('status') == 'running':
            return jsonify({'success': False, 'error': 'A job is already running.'})

    data_path = data.get('data_path', DEFAULT_DATA_PATH)
    save_registration_params(data_path, method=method, preprocessing=preproc_name, loss=loss)

    _PREPROC = {
        'wavelet_denoise': registration.wavelet_denoise,
        'threshold':       registration.threshold,
        'DOG_filter':      registration.DOG_filter,
    }
    preproc_func = _PREPROC.get(preproc_name)
    kwargs       = {'loss': loss} if method == 'elastix' else {}

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running', 'progress': 0,
        'total': len(tif_paths), 'log': [],
    }

    def background_job():
        from collections import defaultdict
        try:
            # Group TIFs by position folder for readable log output
            by_pos = defaultdict(list)
            for p in tif_paths:
                by_pos[Path(p).parents[3]].append(Path(p))

            processed = 0
            for pos_path, tifs in sorted(by_pos.items()):
                tasks[task_id]['log'].append(f'── {pos_path.parent.name} / {pos_path.name}  ({len(tifs)} cells)')

                for tif_path in sorted(tifs):
                    tasks[task_id]['progress'] = processed

                    registration.run_registration([tif_path], method, preproc_func, **kwargs)

                    tasks[task_id]['log'].append(f'{tif_path.parent.name}  saved')
                    processed += 1

            tasks[task_id]['progress'] = len(tif_paths)
            tasks[task_id]['status']   = 'done'
            tasks[task_id]['log'].append('All done!')

        except Exception as e:
            import traceback
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error']  = str(e)
            tasks[task_id]['log'].append(f'ERROR: {e}')
            tasks[task_id]['log'].append(traceback.format_exc())

    threading.Thread(target=background_job, daemon=True).start()
    return jsonify({'success': True, 'task_id': task_id, 'yaml_saved': True})


##############################################################################
# Step 5 – Foci Detection
##############################################################################

@app.route('/step5')
def step5():
    return render_template('step5_detection.html', default_data_path=DEFAULT_DATA_PATH)


# Foci-detection helpers
def _build_detector_params(method, data):
    """Extract detector constructor kwargs from a request data dict."""
    if method == 'Wavelets':
        return {'K':           int(float(data.get('K',           3))),
                'factor':      float(data.get('factor',          4.0)),
                'start_scale': int(float(data.get('start_scale', 0)))}
    if method == 'LOG':
        return {'sigma': float(data.get('sigma', 0.8))}
    if method == 'Hessian':
        return {'sigma': float(data.get('sigma', 0.5))}
    if method == 'TopHat':
        return {'sigma':  float(data.get('sigma',  0.6)),
                'radius': float(data.get('radius', 6.0))}
    if method == 'HDome':
        return {'h': float(data.get('h', 50))}
    if method == 'HDome-smal':
        return {'sigma': float(data.get('sigma', 1.2)),
                'h':     float(data.get('h',     80)),
                's':     float(data.get('s',     1.5))}
    if method == 'MPHD':
        return {'sigma':  float(data.get('sigma',  0.5)),
                'h_init': int(float(data.get('h_init', 5))),
                'R':      int(float(data.get('R',      10)))}
    if method == 'Spotiflow':
        return {'model_name': 'general'}
    return {}


def _make_detection_overlay(frame, coords):
    """Return a base64 PNG of *frame* with detected spot positions overlaid."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=90)
    lo, hi = float(frame.min()), float(frame.max())
    normed = (frame.astype(float) - lo) / (hi - lo) if hi > lo else np.zeros_like(frame, dtype=float)
    ax.imshow(normed, cmap='gray', vmin=0, vmax=1)
    if len(coords) > 0:
        ax.scatter(coords[:, 0], coords[:, 1],
                   facecolors='none', edgecolors='cyan', s=40, linewidths=1.0)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=90)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route('/api/preview_detection', methods=['POST'])
def preview_detection():
    data      = request.get_json()
    reg_paths = data.get('reg_paths', [])
    method    = data.get('method', 'TopHat')
    threshold = float(data.get('threshold', 36))

    if not reg_paths:
        return jsonify({'success': False,
                        'error': 'No registered stacks found. Click "Discover" first.'})
    try:
        params   = _build_detector_params(method, data)
        detector, _ = foci_detection.create_detector(method, params=params, threshold=threshold)

        # pick up to 4 random cells
        n_cells  = min(4, len(reg_paths))
        idxs     = np.random.choice(len(reg_paths), n_cells, replace=False)

        result_cells = []
        for idx in idxs:
            tif_path  = Path(reg_paths[int(idx)])
            stack     = skimage.io.imread(str(tif_path))
            frame_idx = np.random.randint(len(stack))
            frame     = stack[frame_idx]

            dets, enh_stack, _ = foci_detection.run_detector_stack([frame], detector, threshold, return_segmentation=False)
            coords = dets[0]   # shape (N, 4): x, y, t, intensity

            result_cells.append({
                'raw':      array_to_b64png(frame),
                'overlay':  _make_detection_overlay(frame, coords),
                'enhanced': array_to_b64png(enh_stack[0]),
                'n_foci':   len(coords),
                'frame':    int(frame_idx),
                'cell':     tif_path.parent.name,
            })

        return jsonify({'success': True, 'cells': result_cells})

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


@app.route('/api/run_detection', methods=['POST'])
def run_detection():
    data = request.get_json()
    reg_paths = data.get('reg_paths', [])
    method = data.get('method', 'TopHat')
    threshold = float(data.get('threshold', 36))
    return_seg = bool(data.get('return_segmentation', False))

    if not reg_paths:
        return jsonify({'success': False, 'error': 'No registered stacks to process.'})

    for t in tasks.values():
        if t.get('status') == 'running':
            return jsonify({'success': False, 'error': 'A job is already running.'})

    params = _build_detector_params(method, data)
    data_path = data.get('data_path', DEFAULT_DATA_PATH)
    save_detection_params(data_path, method=method, threshold=threshold, return_segmentation=return_seg, params=params)
    
    task_id  = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running', 'progress': 0,
        'total': len(reg_paths), 'log': [],
    }

    def background_job():
        from collections import defaultdict
        try:
            detector, _ = foci_detection.create_detector(method, params=params, threshold=threshold)

            by_pos = defaultdict(list)
            for p in reg_paths:
                by_pos[Path(p).parents[2]].append(Path(p))

            processed = 0
            for pos_path, tifs in sorted(by_pos.items()):
                tasks[task_id]['log'].append(
                    f'── {pos_path.parent.name} / {pos_path.name}  ({len(tifs)} cells)')

                for tif_path in sorted(tifs):
                    tasks[task_id]['progress'] = processed

                    foci_detection.run_detection([tif_path], detector, threshold, return_segmentation=return_seg)

                    tasks[task_id]['log'].append(f'{tif_path.parent.name}  saved')
                    processed += 1

            tasks[task_id]['progress'] = len(reg_paths)
            tasks[task_id]['status']   = 'done'
            tasks[task_id]['log'].append('All done!')

        except Exception as e:
            import traceback
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error']  = str(e)
            tasks[task_id]['log'].append(f'ERROR: {e}')
            tasks[task_id]['log'].append(traceback.format_exc())

    threading.Thread(target=background_job, daemon=True).start()
    return jsonify({'success': True, 'task_id': task_id, 'yaml_saved': True})


##############################################################################
# Step 6 – Foci Tracking
##############################################################################

@app.route('/step6')
def step6():
    return render_template('step6_foci_tracking.html', default_data_path=DEFAULT_DATA_PATH)


@app.route('/api/preview_foci_tracking', methods=['POST'])
def preview_foci_tracking():
    data      = request.get_json()
    reg_paths = data.get('reg_paths', [])
    method    = data.get('tracking_method', 'GNN')
    min_len   = int(data.get('min_track_length', 3))
    max_dist  = float(data.get('max_distance', 5.0))
    gap_close = int(data.get('gap_closing', 2))
    use_seg   = bool(data.get('use_segmentation', True))

    if not reg_paths:
        return jsonify({'success': False,
                        'error': 'No registered stacks found. Click "Discover" first.'})
    try:
        disp_path = Path(reg_paths[np.random.randint(len(reg_paths))])

        foci_trackers.run_foci_tracker([disp_path], method=method, min_track_length=min_len, max_distance=max_dist,
            gap_closing=gap_close, use_segmentation=use_seg)

        # generate movie exactly as in the notebook
        plt_figures.animate_from_xml(disp_path, dpi=240, interval=20, fps=7, lw=1.0, ms=1.0)

        movie_path = disp_path.parent / 'tracking_movie.mp4'
        if not movie_path.exists():
            return jsonify({'success': False,
                            'error': 'Movie file was not created (is ffmpeg installed?).'})

        with open(movie_path, 'rb') as f:video_b64 = base64.b64encode(f.read()).decode('utf-8')

        # count tracks from xml
        xml_path = disp_path.parent / 'tracks.xml'
        n_tracks = 0
        if xml_path.exists():
            tracks_arr = plt_figures.tracks_from_isbi_xml(str(xml_path))
            n_tracks = int(len(np.unique(tracks_arr[:, 0]))) if len(tracks_arr) else 0

        return jsonify({
            'success':       True,
            'movie_b64':     video_b64,
            'selected_path': str(disp_path),
            'n_tracks':      n_tracks,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


@app.route('/api/run_foci_tracking', methods=['POST'])
def run_foci_tracking():
    data      = request.get_json()
    reg_paths = data.get('reg_paths', [])
    method    = data.get('tracking_method', 'GNN')
    min_len   = int(data.get('min_track_length', 3))
    max_dist  = float(data.get('max_distance', 5.0))
    gap_close = int(data.get('gap_closing', 2))
    use_seg   = bool(data.get('use_segmentation', True))

    if not reg_paths:
        return jsonify({'success': False, 'error': 'No registered stacks to process.'})

    for t in tasks.values():
        if t.get('status') == 'running':
            return jsonify({'success': False, 'error': 'A job is already running.'})

    data_path = data.get('data_path', DEFAULT_DATA_PATH)
    save_foci_tracking_params(data_path, method=method, min_track_length=min_len, max_distance=max_dist,
        gap_closing=gap_close, use_segmentation=use_seg)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'running', 'progress': 0,
        'total': len(reg_paths), 'log': [],
    }

    def background_job():
        from collections import defaultdict
        try:
            by_pos = defaultdict(list)
            for p in reg_paths:
                by_pos[Path(p).parents[2]].append(Path(p))

            processed = 0
            for pos_path, tifs in sorted(by_pos.items()):
                tasks[task_id]['log'].append(
                    f'── {pos_path.parent.name} / {pos_path.name}  ({len(tifs)} cells)')

                for tif_path in sorted(tifs):
                    tasks[task_id]['progress'] = processed

                    foci_trackers.run_foci_tracker([tif_path], method=method, min_track_length=min_len,
                        max_distance=max_dist, gap_closing=gap_close, use_segmentation=use_seg)

                    xml_path = tif_path.parent / 'tracks.xml'
                    n_tracks = 0
                    if xml_path.exists():
                        arr = plt_figures.tracks_from_isbi_xml(str(xml_path))
                        n_tracks = int(len(np.unique(arr[:, 0]))) if len(arr) else 0

                    tasks[task_id]['log'].append(f'{tif_path.parent.name}  {n_tracks} tracks saved')
                    processed += 1

            tasks[task_id]['progress'] = len(reg_paths)
            tasks[task_id]['status']   = 'done'
            tasks[task_id]['log'].append('All done!')

        except Exception as e:
            import traceback
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['error']  = str(e)
            tasks[task_id]['log'].append(f'ERROR: {e}')
            tasks[task_id]['log'].append(traceback.format_exc())

    threading.Thread(target=background_job, daemon=True).start()
    return jsonify({'success': True, 'task_id': task_id, 'yaml_saved': True})


##############################################################################
# Plot helpers
##############################################################################

def _fig_to_b64(fig, dpi=150):
    """Encode a matplotlib figure as a base64 PNG string and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


##############################################################################
# Step 7 – Plot results
##############################################################################

@app.route('/step7')
def step7():
    return render_template('step7_results.html',
                           default_data_path=DEFAULT_DATA_PATH)


@app.route('/api/discover_conditions', methods=['POST'])
def discover_conditions():
    data      = request.get_json()
    data_path = Path(data.get('data_path', DEFAULT_DATA_PATH))
    TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    try:
        conds = sorted([d for d in data_path.iterdir() if d.is_dir()])
        return jsonify({
            'success': True,
            'conditions': [
                {'tag': d.name, 'name': d.name, 'color': TAB10[i % len(TAB10)]}
                for i, d in enumerate(conds)
            ],
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/generate_plots', methods=['POST'])
def generate_plots():
    from lift.general_utils import plot_utils
    from lift.general_utils.read_xml import extract_tracking_info, extract_tracking_features

    data         = request.get_json()
    data_path    = Path(data.get('data_path', DEFAULT_DATA_PATH))
    conditions   = data.get('conditions', [])
    num_frames   = int(data.get('num_frames', 288))
    dt           = float(data.get('dt', 5))
    time_offset  = float(data.get('time_offset', 0))
    remove_final = int(data.get('remove_final_frames', 0))
    error_band   = data.get('error_band', 'sem') or None
    foci_max     = float(data.get('foci_max', 25))
    max_bin      = int(data.get('max_bin', 300))
    percent_max  = float(data.get('percent_max', 13))
    track_filter = data.get('track_filter', 'no filter')

    if not conditions:
        return jsonify({'success': False, 'error': 'No conditions selected.'})

    try:
        plot_samples = []
        skipped = []
        for cond in conditions:
            tag   = cond['tag']
            paths = [Path(p) for p in data_path.glob(f"{tag}/*/Pos*/results/registered/*")]
            if not paths:
                skipped.append(tag)
                continue
            TL, AT, _ = extract_tracking_info(paths, num_frames, dt,
                                              track_filter=track_filter)
            intensity_arr, size_arr, mean_int_arr = extract_tracking_features(paths, num_frames,
                                                                              track_filter=track_filter)
            color = cond.get('color', '#1f77b4')
            fill  = cond.get('fill', False)
            plot_samples.append({
                'name':          cond.get('name', tag),
                'active_tracks': AT,
                'track_lengths': TL,
                'intensity':     intensity_arr,
                'size':          size_arr,
                'mean_intensity': mean_int_arr,
                'color':         color,
                'linestyle':     cond.get('linestyle', 'solid'),
                'alpha':         float(cond.get('alpha', 0.8)),
                'facecolor':     color if fill else 'none',
                'edgecolor':     'none' if fill else color,
                'hatch':         cond.get('hatch', ''),
            })

        if not plot_samples:
            return jsonify({'success': False,
                            'error': 'No tracking data found for the selected conditions.'})

        shifted_time = np.arange(num_frames) * dt / 60 + time_offset / 60
        bins = np.arange(0, max_bin, dt)

        # active tracks over time 
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        plot_utils.plot_AT(shifted_time, plot_samples, remove_final_frames=remove_final, error_band=error_band, ax=ax1)
        ax1.set_xlim(0, num_frames * dt / 60)
        ax1.set_ylim(0, foci_max)
        fig1.tight_layout()
        png_at = _fig_to_b64(fig1)

        # dwell-time histogram (linear) 
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        plot_utils.plot_hist(bins, plot_samples, log=False, ax=ax2)
        ax2.set_ylim(0, percent_max / 100)
        fig2.tight_layout()
        png_hist = _fig_to_b64(fig2)

        # violin
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        plot_utils.plot_violin(plot_samples, max_bin, ax=ax3)
        fig3.tight_layout()
        png_violin = _fig_to_b64(fig3)

        # feature plots (only when enriched XML data is available)
        has_features = any(s.get('mean_intensity') is not None for s in plot_samples)
        png_mean_intensity = png_size = None
        if has_features:
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            plot_utils.plot_feature('mean_intensity', shifted_time, plot_samples,
                                    ylabel='Mean spot intensity [a.u.]',
                                    remove_final_frames=remove_final,
                                    error_band=error_band, ax=ax4)
            ax4.set_xlim(0, num_frames * dt / 60)
            fig4.tight_layout()
            png_mean_intensity = _fig_to_b64(fig4)

            fig5, ax5 = plt.subplots(figsize=(7, 4))
            plot_utils.plot_feature('size', shifted_time, plot_samples,
                                    ylabel='Mean spot area [px]',
                                    remove_final_frames=remove_final,
                                    error_band=error_band, ax=ax5)
            ax5.set_xlim(0, num_frames * dt / 60)
            fig5.tight_layout()
            png_size = _fig_to_b64(fig5)

        return jsonify({
            'success':           True,
            'png_at':            png_at,
            'png_hist':          png_hist,
            'png_violin':        png_violin,
            'png_mean_intensity': png_mean_intensity,
            'png_size':          png_size,
            'has_features':      has_features,
            'n_conditions':      len(plot_samples),
            'skipped':           skipped,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


@app.route('/api/run_stats', methods=['POST'])
def run_stats():
    from lift.general_utils import plot_utils
    from lift.general_utils.read_xml import extract_tracking_info

    data         = request.get_json()
    data_path    = Path(data.get('data_path', DEFAULT_DATA_PATH))
    conditions   = data.get('conditions', [])
    num_frames   = int(data.get('num_frames', 288))
    dt           = float(data.get('dt', 5))
    track_filter = data.get('track_filter', 'no filter')
    n_resamples  = int(data.get('n_resamples', 5000))

    if not conditions:
        return jsonify({'success': False, 'error': 'No conditions selected.'})

    try:
        plot_samples = []
        for cond in conditions:
            tag   = cond['tag']
            paths = [Path(p) for p in data_path.glob(f"{tag}/*/Pos*/results/registered/*")]
            if not paths:
                continue
            TL, _, _ = extract_tracking_info(paths, num_frames, dt,
                                             track_filter=track_filter)
            plot_samples.append({'name': cond.get('name', tag), 'track_lengths': TL})

        if not plot_samples:
            return jsonify({'success': False,
                            'error': 'No tracking data found for the selected conditions.'})

        stats_text = plot_utils.run_statistical_tests(
            plot_samples, n_resamples=n_resamples)

        return jsonify({'success': True, 'stats_text': stats_text})

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'traceback': traceback.format_exc()})


##############################################################################

if __name__ == '__main__':

    app.run(debug=False, port=5000, use_reloader=False, threaded=True)
