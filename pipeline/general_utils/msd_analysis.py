import numpy as np
from pathlib import Path
from xml.dom import minidom


def analyse_diffusion(paths, dt, pixelsize=1.0, min_track_length=5, max_lag_fraction=0.4):
    """
    Compute per-track diffusion constant D and anomalous exponent alpha from tracks.xml.

    Fits MSD(tau) = 4 * D * tau^alpha in log-log space for each track.
    Coordinates in the XML are in pixels; multiply by pixelsize to get MSD in um^2.
    Lag times are in seconds. Gapped tracks (from gap-closing) are handled correctly —
    only pairs with both frames present contribute to each lag.

    Parameters
    ----------
    paths : list of Path-like
        Paths to per-nucleus directories (same list passed to extract_tracking_info).
        Each directory must contain a tracks.xml file.
    dt : float
        Time between frames in seconds.
    pixelsize : float
        Pixel size in um/pixel. Set to 1.0 to keep MSD in pixels^2.
    min_track_length : int
        Minimum number of detections a track must have to be included.
    max_lag_fraction : float
        Fraction of track length used as the maximum lag for the MSD fit (default 0.4).
        Keeps long-lag estimates from becoming noisy due to few data points.

    Returns
    -------
    D_values : ndarray, shape (n_tracks,)
        Generalised diffusion coefficient per track in um^2 / s^alpha.
    alpha_values : ndarray, shape (n_tracks,)
        Anomalous exponent per track (alpha=1: normal diffusion, <1: subdiffusion).
    """
    D_values = []
    alpha_values = []

    for nuc_dir in paths:
        nuc_dir = Path(nuc_dir)
        xml_path = nuc_dir / "tracks.xml"
        if not xml_path.exists():
            continue

        tree = minidom.parse(str(xml_path))
        for pt in tree.getElementsByTagName('particle'):
            detections = pt.getElementsByTagName('detection')
            if len(detections) < min_track_length:
                continue

            frame_to_pos = {}
            for det in detections:
                t = int(det.getAttribute('t'))
                x = float(det.getAttribute('x')) * pixelsize
                y = float(det.getAttribute('y')) * pixelsize
                frame_to_pos[t] = (x, y)

            frames = sorted(frame_to_pos)
            n = len(frames)
            max_lag = max(2, int(n * max_lag_fraction))

            msd_vals = []
            lag_times = []
            for lag in range(1, max_lag + 1):
                displacements_sq = []
                for f in frames:
                    if f + lag in frame_to_pos:
                        dx = frame_to_pos[f + lag][0] - frame_to_pos[f][0]
                        dy = frame_to_pos[f + lag][1] - frame_to_pos[f][1]
                        displacements_sq.append(dx**2 + dy**2)
                if len(displacements_sq) < 2:
                    continue
                msd_vals.append(np.mean(displacements_sq))
                lag_times.append(lag * dt)

            if len(lag_times) < 2:
                continue

            msd_arr = np.array(msd_vals)
            lag_arr = np.array(lag_times)
            valid = msd_arr > 0
            if valid.sum() < 2:
                continue

            log_tau = np.log(lag_arr[valid])
            log_msd = np.log(msd_arr[valid])
            alpha, log_4D = np.polyfit(log_tau, log_msd, 1)
            D = np.exp(log_4D) / 4.0

            D_values.append(D)
            alpha_values.append(alpha)

    return np.array(D_values), np.array(alpha_values)
