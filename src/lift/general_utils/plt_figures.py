from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skimage.io
import skimage.measure
from xml.dom import minidom


def animate_from_coords(tracks, images, movie_loc, figsize=(6,6), dpi=80, interval=50, fps=5, vmax=None, lw=2.0, ms=3.0):
    """
    Animate tracking from existing coordinates.

    Parameters
    ----------
    tracks : np.ndarray
        Array of shape (N, 4): columns [track_id, frame, y, x].
    img_files : list of Path or str
        Raw images to display.
    movie_loc : str
        Path to save the output mp4 file.
    figsize : tuple
        Figure size.
    dpi : int
        Figure DPI.
    interval : integer
        Interval between frames for displaying in the GUI only (not for saved .mp4 file)
    fps: integer
        number of frames per second for the saved .mp4 file
    vmax : float
        Optional max intensity for raw images.
    lw: float
        linewidth used for displaying tracks
    ms: float
        size of the marker used for displaying tracks
    """
    track_ids = np.unique(tracks[:,0]).astype(int)
    tracks_by_id = {tid: tracks[tracks[:,0] == tid] for tid in track_ids}

    # Colors
    cmap = plt.get_cmap("tab20")
    colors = {tid: cmap(i % 20) for i, tid in enumerate(track_ids)}

    # Figure setup
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(images[0], cmap="gray", vmax=vmax)
    ax.axis("off")

    # Create lines and points for each track
    lines = {}
    points = {}
    for tid in track_ids:
        line, = ax.plot([], [], lw=lw, color=colors[tid])
        point, = ax.plot([], [], "o", color=colors[tid], markersize=ms)
        lines[tid] = line
        points[tid] = point

    # Update function
    def update(frame):
        im.set_data(images[frame])
        for tid in track_ids:
            track = tracks_by_id[tid]
            start = track[:,1].min()
            end = track[:,1].max()

            # Hide tracks not active in this frame
            if frame < start or frame > end:
                lines[tid].set_data([], [])
                points[tid].set_data([], [])
                continue

            visible = track[track[:,1] <= frame]
            xs = visible[:,3]
            ys = visible[:,2]

            lines[tid].set_data(xs, ys)
            points[tid].set_data([xs[-1]], [ys[-1]])

        ax.set_title(f"Frame {frame}")
        return [im] + list(lines.values()) + list(points.values())

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(images), interval=interval, blit=False)
    ani.save(movie_loc, dpi=dpi, fps=fps)
    plt.close(fig)


def tracks_from_isbi_xml(xml_file):
    """
    Parse ISBI particle tracking XML and return tracks array
    with columns [track_id, frame, y, x].
    """

    dom = minidom.parse(xml_file)
    particles = dom.getElementsByTagName("particle")

    rows = []
    for tid, particle in enumerate(particles):

        detections = particle.getElementsByTagName("detection")

        for det in detections:
            frame = int(det.getAttribute("t"))
            x = float(det.getAttribute("x"))
            y = float(det.getAttribute("y"))

            rows.append((tid, frame, y, x))

    return np.array(rows)


def animate_from_xml(tif_path, figsize=(6,6), dpi=80, interval=20, fps=7, vmax=None, lw=1.0, ms=2.0):
    """
    Animate ISBI particle tracking results from XML file.
    """

    tif_path = Path(tif_path)
    xml_file = str(tif_path.parent / "tracks.xml")

    image_stack = skimage.io.imread(str(tif_path))
    movie_loc = tif_path.parent / "tracking_movie.mp4"  # location for saving

    # load the tracks
    tracks = tracks_from_isbi_xml(xml_file)

    animate_from_coords(tracks, image_stack, movie_loc, figsize=figsize, dpi=dpi, interval=interval, fps=fps, vmax=vmax, lw=lw, ms=ms)


def animate_from_masks(base_path, figsize=(6,6), dpi=80, interval=50, fps=10, vmax=None, lw=2.0, ms=3.0):
    """
    Animate tracking from masks as output in the ctc format images by first extracting centroids,
    then calling animate_from_coords.

    parameters:
    base_path : path to the results folder for the experiment
    Other parameters are forwarded to animate_from_coords.
    """

    tracking_path = base_path / "cell_tracking"  # location for the tracking result
    mask_files = sorted((tracking_path).glob("t*.tif"))  # get the masks
    img_files = sorted((base_path.parent / "raw").glob("*.tif"))  # get the raw images
    movie_loc = tracking_path / "tracking.mp4"  # location for saving the movie
    images = np.stack([skimage.io.imread(f) for f in img_files])
    
    # Extract centroids from masks
    rows = []
    for frame, mask_file in enumerate(mask_files):
        mask = skimage.io.imread(mask_file)
        for r in skimage.measure.regionprops(mask):
            y, x = r.centroid
            rows.append((r.label, frame, y, x))
    tracks = np.array(rows)  # columns: track_id, frame, y, x

    # Call the generic coordinate-based animation
    animate_from_coords(tracks, images, movie_loc, figsize=figsize, dpi=dpi, interval=interval, fps=fps, vmax=vmax, lw=lw, ms=ms)
    