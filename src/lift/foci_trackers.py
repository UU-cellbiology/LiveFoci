from pathlib import Path
import os
import shutil
import numpy as np
from scipy.optimize import linear_sum_assignment
import subprocess
from tifffile import imread
from xml.dom import minidom

#####
#
# below are the helper functions to run the different tracking algorithms on the .tif files and save the resulting output
#
#####


def run_foci_tracker(path_list, method, **kwargs):
    """
    performs foci tracking using the specified tracking method 
    
    path_list: should be a list of paths to the .tif files that need to be tracked. 

    method: string specifying which tracking method to use. this tracking method takes the .tif file
    and the 'detected_foci.txt' (containing the detected foci coordinates) as input to perform the 
    linking between the detected foci

    optinal parameters:
    max_distance: the maximum allowed distance for linking detections [pixels]
    gap_close: number of frames to close between tracks when detections are missing
    min_length: the minimum allowed length of the tracks defined as minimum number of detections
    use_segmentation: for Trackastra only, True for using the segmented objects to calculate features 
                        and False for using detected coordinates only
    """
    if method is None:
        from lift._helpers import _detect_foci_tracker
        method = _detect_foci_tracker()

    print(f"foci tracking method:     {method}\n")

    save_features = kwargs.pop('save_features', True)

    if method=='GNN':
        max_dist = kwargs.pop("max_distance", 5.0)
        gap_close = kwargs.pop("gap_closing", 2)
        min_length = kwargs.pop("min_track_length", 3)

        tracker = GNN_tracker(max_distance=max_dist, gap_closing=gap_close, min_track_length=min_length)
        tracker_func = tracker.track
    elif method=='NGMA':
        max_dist = kwargs.pop("max_distance", 5.0)
        gap_close = kwargs.pop("gap_closing", 3)
        min_length = kwargs.pop("min_track_length", 3)

        tracker = NGMA_track(min_track_length=min_length, max_distance=max_dist, gap_closing=gap_close)
        tracker_func = tracker.forward
    elif method=='trackastra':
        use_seg = kwargs.pop("use_segmentation", True)
        min_length = kwargs.pop("min_track_length", 3)

        tracker = trackastra_tracker(min_track_length=min_length, use_segmentations=use_seg)
        tracker_func = tracker.track
    else:
        raise ValueError(f"Unknown foci tracking method: {method!r}")

    for path in path_list:
        path = Path(path)
        tracker_func(path)

        if save_features:
            dir_path = path.parent
            xml_path  = dir_path / "tracks.xml"
            det_txt   = dir_path / "detected_foci.txt"
            if xml_path.exists() and det_txt.exists():
                seg_tif = dir_path / "spot_segmentation.tif"
                enrich_tracks_xml(xml_path, det_txt,
                                  seg_tif if seg_tif.exists() else None)

#####
#
# General function to save the tracks to the required xml format from the particle tracking challange.
# the tracks as input are given as a list of tracks[track_id][y,x,t]
#
#####

def write_isbi_xml(tracks, out_file):        
    doc = minidom.Document()
    
    # Root element
    root = doc.createElement("root")
    doc.appendChild(root)
    
    # ISBI contest element
    contest = doc.createElement("TrackContestISBI2012")
    contest.setAttribute("scenario", "foci_tracking")
    contest.setAttribute("snr", "unknown")
    contest.setAttribute("density", "unknown")
    root.appendChild(contest)
    
    # Tracks
    for tr in tracks:
        particle = doc.createElement("particle")
        contest.appendChild(particle)
    
        for x, y, t in tr:
            det = doc.createElement("detection")
            det.setAttribute("t", str(int(t)))
            det.setAttribute("x", f"{x:.2f}")
            det.setAttribute("y", f"{y:.2f}")
            det.setAttribute("z", f"{0.0:.2f}")  
            particle.appendChild(det)
    
    xml_str = doc.toprettyxml(indent="  ", encoding="utf-8")
    
    with open(out_file, "wb") as f:
        f.write(xml_str)


def enrich_tracks_xml(xml_path, det_txt_path, seg_tif_path=None):
    """
    Post-process tracks.xml to add intensity, size, and mean_intensity attributes
    to each <detection> by matching (t, x, y) back to detected_foci.txt.

    When detected_foci.txt was saved with return_segmentation=True it already contains
    size_px (col 4) and mean_intensity (col 5); otherwise those are computed from
    spot_segmentation.tif if available.  Pass save_features=False to run_foci_tracker
    to skip this step entirely (useful for benchmarking runs).
    """
    det_data = np.loadtxt(str(det_txt_path))
    if det_data.size == 0:
        return
    if det_data.ndim == 1:
        det_data = det_data.reshape(1, -1)

    has_seg_cols = det_data.shape[1] >= 6

    # build lookup: (t_int, x_int, y_int) -> feature dict
    lookup = {}
    for row in det_data:
        key = (int(round(row[2])), int(round(row[0])), int(round(row[1])))
        entry = {'intensity': float(row[3])}
        if has_seg_cols:
            entry['size'] = float(row[4])
            entry['mean_intensity'] = float(row[5])
        lookup[key] = entry

    # if txt has only 4 cols but a seg tif exists, compute size/mean_intensity from it
    if not has_seg_cols and seg_tif_path is not None:
        import skimage.measure
        seg_stack = imread(str(seg_tif_path))
        tif_files = sorted(Path(det_txt_path).parent.glob("I_*.tif"))
        if tif_files:
            raw_stack = imread(str(tif_files[0]))
            for row in det_data:
                t_int = int(round(row[2]))
                x_int = int(round(row[0]))
                y_int = int(round(row[1]))
                key = (t_int, x_int, y_int)
                if t_int < len(seg_stack):
                    props = skimage.measure.regionprops(seg_stack[t_int],
                                                        intensity_image=raw_stack[t_int])
                    for prop in props:
                        cy, cx = prop.centroid
                        if abs(cy - y_int) < 1.5 and abs(cx - x_int) < 1.5:
                            lookup[key]['size'] = float(prop.area)
                            lookup[key]['mean_intensity'] = float(prop.mean_intensity)
                            break

    # parse and enrich the XML
    tree = minidom.parse(str(xml_path))
    n_missed = 0
    for det in tree.getElementsByTagName('detection'):
        t = int(det.getAttribute('t'))
        x = int(round(float(det.getAttribute('x'))))
        y = int(round(float(det.getAttribute('y'))))
        entry = lookup.get((t, x, y))
        if entry:
            det.setAttribute('intensity', f"{entry['intensity']:.2f}")
            if 'size' in entry:
                det.setAttribute('size', f"{entry['size']:.1f}")
                det.setAttribute('mean_intensity', f"{entry['mean_intensity']:.2f}")
        else:
            n_missed += 1

    if n_missed:
        print(f"  Warning: {n_missed} detections in {Path(xml_path).name} "
              f"could not be matched to {Path(det_txt_path).name}")

    features_path = Path(xml_path).parent / "tracks_features.xml"
    with open(features_path, 'wb') as f:
        f.write(tree.toxml(encoding='utf-8'))

#####
#
# Below are the classes for the different tracking algorithms
#
#####


class GNN_tracker(object):

    """
    Global nearest neighbour tracking algorithm
    """
    
    def __init__(self, max_distance=5.0, gap_closing=3, min_track_length=4):
        super().__init__()

        self.max_dist = max_distance
        self.gap_close = gap_closing
        self.min_length = min_track_length              

    def build_cost_matrix(self, prev_dets, curr_dets):
        """
        build the cost matrix as described in: Khuloud Jaqaman, Dinah Loerke, Marcel Mettlen, Hirotaka Kuwata, Sergio Grinstein, Sandra L Schmid, and Gaudenz
        Danuser. Robust single-particle tracking in live-cell time-lapse sequences. Nature methods, 2008.
        """
        
        death_cost = 1000   # Cost of terminating a track
        birth_cost = 1000    # Cost of starting a new track
        dummy_cost = 100  # should be smaller than birth and death cost
    
        n_prev = len(prev_dets)
        n_curr = len(curr_dets)
    
        size = n_prev + n_curr
    
        cost = np.full((size, size), 1e6)
    
        # block with eucledian distances
        if n_prev > 0 and n_curr > 0:
            dist = np.linalg.norm(prev_dets[:, None, :2] - curr_dets[None, :, :2], axis=2)
            dist[dist > self.max_dist] = 1e6
    
            cost[:n_prev, :n_curr] = dist
    
        # death block (diagonal)
        for i in range(n_prev):
            cost[i, n_curr + i] = death_cost
    
        # Birth block (diagonal)
        for j in range(n_curr):
            cost[n_prev + j, j] = birth_cost
    
        # Dummy block (lower right)
        cost[n_prev:, n_curr:] = dummy_cost
    
        return cost

    
    def build_links(self, detections, num_frames):
        """
        form tracks for every frame t and t+1 for all of the num_frames amount of frames
        the detections are a numpy array stored as (x, y, t) and a different row for every detection
        """  
        tracks = []
        active_tracks = []
        
        for t in range(num_frames):
        
            curr_dets = detections[detections[:,2] == t, :3]
        
            prev_dets = np.array([tracks[idx][-1] for idx in active_tracks]) if active_tracks else np.empty((0,3))
        
            cost = self.build_cost_matrix(prev_dets, curr_dets)
        
            row_ind, col_ind = linear_sum_assignment(cost)
        
        
            new_active_tracks = []
            for r, c in zip(row_ind, col_ind):
        
                # link to existing track
                if r < len(prev_dets) and c < len(curr_dets):
                    track_idx = active_tracks[r]
                    tracks[track_idx].append(curr_dets[c])
                    new_active_tracks.append(track_idx)
        
                # birth event
                elif r >= len(prev_dets) and c < len(curr_dets):
                    tracks.append([curr_dets[c]])
                    new_active_tracks.append(len(tracks)-1)
        
                # death event
                elif r < len(prev_dets) and c >= len(curr_dets):
                    pass
        
            active_tracks = new_active_tracks
    
        return tracks


    def gap_close_iterative(self, tracks):
        """
        iterativly close all the gaps between separate tracks until all gaps are connected
        """
    
        while True:
    
            tracks, merged_any = self.gap_close_tracks(tracks, self.max_dist, self.gap_close)
    
            if not merged_any:
                break
    
        return tracks

    
    @staticmethod
    def gap_close_tracks(tracks, max_distance, max_gap):
        """
        close gaps between separate tracks. It will only connect the end of one track to the start of the other track
        and remove the track later in time. To be able to connect the same track to another track again this functions needs
        to be run multiple times

        input: 
        tracks: the tracks stored as track[track id][x, y, t]
        max_distance: maximum eucledian distance to allow two tracks to be linked together
        max_gap: number of frames to close between tracks
        """
    
        merged = [False] * len(tracks)
        
        starts = []
        ends = []    
        for i, track in enumerate(tracks):
        
            starts.append({
                "track": i,
                "t": track[0][2],
                "pos": np.array(track[0][:2])
            })
        
            ends.append({
                "track": i,
                "t": track[-1][2],
                "pos": np.array(track[-1][:2])
            })
        
        starts.sort(key = lambda x: x["t"])
        ends.sort(key   = lambda x: x["t"])
        
        merged_any = False
        for end in ends:
        
            if merged[end["track"]]:
                continue
        
            best_match = None
            best_dist = np.inf
            for start in starts:
        
                if merged[start["track"]]:  # skip tracks that we already merged to prevent issues
                    continue
        
                # skip if the track id is the same
                if start["track"] == end["track"]:
                    continue
        
                dt = start["t"] - end["t"]
        
                # only link to starts in the future
                if dt <= 0:
                    continue
        
                # as starts are sorted all starts that follow should also be out of the linking range
                if dt > max_gap:
                    break
        
        
                dist = np.linalg.norm(end["pos"] - start["pos"])
        
                if dist <= max_distance and dist < best_dist:
                    best_dist = dist
                    best_match = start
        
        
            if best_match is not None:
                t1 = end["track"]
                t2 = best_match["track"]
        
                tracks[t1].extend(tracks[t2])
                merged[t2] = True
                merged_any = True
        
        # keep the new tracks without the merged tracks
        new_tracks = [tracks[i] for i in range(len(tracks)) if not merged[i]]
    
        return new_tracks, merged_any

    
    @staticmethod
    def load_detection_file(det_path):

        if det_path.stat().st_size == 0: # catch a case with no detections  
            return np.empty((0, 4))
        
        detections = np.loadtxt(det_path)
            
        if detections.ndim == 1:  # rehsape to 2d array when we have only 1 detection
            detections = detections[None, :]

        return detections

    
    def track(self, tif_path):
        
        dir_path = Path(tif_path).parent
        detection_path = dir_path / "detected_foci.txt"  # get the path for the detections.txt

        num_frames = imread(tif_path).shape[0]  
        detections = self.load_detection_file(detection_path) 
                  
        # initial tracks formed by only looking at the next frame
        tracks = self.build_links(detections, num_frames)
        
        # close gaps in the tracks
        tracks = self.gap_close_iterative(tracks)
        
        # filter out short tracks 
        if self.min_length > 1:
            tracks = [trk for trk in tracks if len(trk) >= self.min_length]
        
        # save the tracks to an xml file
        out_xml =  dir_path / "tracks.xml"
        write_isbi_xml(tracks, out_xml)



class NGMA_track(object):
    
    def __init__(self, min_track_length, max_distance, gap_closing=3, motion_model=3):
        super().__init__()
        
        # for the NGMA tracker you control the buffer size but gap_closing = buffer size - 2 so it is practically
        # the same parameter
        self.gap = gap_closing  
        self.buffer_size = gap_closing + 2
        self.mm = motion_model   
        
        # was: self.base_dir = Path(__file__).parent.resolve()
        # NGMA_utils no longer ships inside the package — it's downloaded
        # into a local cache on first use instead (see _ensure_binary_utils
        # in _helpers.py).
        from lift._helpers import _ensure_binary_utils
        self.base_dir = _ensure_binary_utils("NGMA_utils")

        if os.name == "nt":  # command for windows
            self.java = self.base_dir / "ImageJ" / "jre" / "bin" / "java.exe"
            self.classpath_sep = ";"
        else:  # command for other systems
            self.java = self.base_dir / "ImageJ" / "jre" / "bin" / "java"
            self.classpath_sep = ":"
                            
        # location of the java tracking plugin
        self.TP_dir = self.base_dir / "SOSTracker commandline"

        # Classpath jars
        jars = ["VENI_.jar", "ij.jar", "imagescience.jar", "Jama-1.0.2.jar"]
        classpath = self.classpath_sep.join(jars)

        self.plugins = ["-Xmx3000m", "-cp", classpath, "ws.smal.sos.eval.linking.Linker"]

        # modify the parameter file to use the user specified parameters        
        self.modify_parameter_file(min_track_length, max_distance)
        
        
    def forward(self, tif_file):

        tif_file = Path(tif_file).resolve()   # turn path into global path
        dir_path = Path(tif_file).parent

        num_frames = imread(tif_file).shape[0]
        
        detection_path = dir_path / "detected_foci.txt"
        self.reformat_detections(detection_path, num_frames)

        # copy parameters.xml file to folder of the image
        shutil.copy(self.TP_dir / "modified_parameters.xml", dir_path / "parameters.xml")

        args = [str(tif_file), str(num_frames), str(self.mm)]

        # final command for the tracking
        cmd = [str(self.java), *self.plugins, *args]

        p = subprocess.run(cmd, capture_output=True, text=True, cwd=self.TP_dir)
        print(p.stderr)
        print(p.stdout)

        # rename the created tracking 
        old_file = dir_path / f"{dir_path.name}.NGMA.xml"
        if old_file.exists():
            new_file = dir_path / "tracks.xml"  
            old_file.replace(new_file)
        else:
            print(f"no tracks found for {dir_path}")

    
    def modify_parameter_file(self, min_track_length, max_distance):    
        param_file = self.TP_dir / "parameters.xml"
        doc = minidom.parse(str(param_file))
        parameters = doc.getElementsByTagName("parameter")
        
        # Loop through parameters and modify the ones you want
        for param in parameters:
            param_id = param.getAttribute("ID")
        
            if param_id == "MinimumTrackLength":
                param.setAttribute("value", f"{min_track_length:.1f}")
        
            if param_id == "gateDistanceDiffusion":
                param.setAttribute("value", f"{max_distance:.1f}")

            if param_id == "LinkingBufferSize":
                param.setAttribute("value", f"{self.buffer_size:.1f}")
        
        # Save to new file
        output_file = self.TP_dir / "modified_parameters.xml"
        with open(output_file, "w") as f:
            doc.writexml(f, indent="  ", addindent="  ", newl="\n")

        
    def reformat_detections(self, path, num_frames): 
        """
        function to add additional detection entries as required for the SOS tracker
        and saves it to the correct file format. 
        path: points to the detected_foci.txt file
        num_frames is the number of frames in the movie
        insert_gap: how often to insert fake detections
        """
        path = Path(path)
        detections = np.loadtxt(path)

        to_add = 10 - detections.shape[1]  # how many entries to add to get 10 in total
        extra_entries = np.zeros((detections.shape[0], to_add))
        detections = np.hstack((detections, extra_entries))
    
        detections = self.add_fake_detections(detections, num_frames, self.buffer_size-1)
    
        save_path = path.parent / "detections.xml.txt"  
        np.savetxt(save_path, detections, fmt='%.2f', delimiter='\t')

    
    @staticmethod
    def add_fake_detections(detections, num_frames, insert_gap):
        """
        function to add additional (fake) detection when there is a number of frames without
        any detections equal to 'insert_gap'. This is required to prevent empty arrays in the
        SOS tracker. num_frames is the total number of frames in the stack.
        """
        # get timepoints with detections
        timepoints = np.unique(detections[:,2]).astype(int)
        
        # looks for gaps with no detections
        missing_mask = np.ones(num_frames, dtype=bool)
        missing_mask[timepoints] = False
        missing_t = np.where(missing_mask)[0]
        
        # look for groups of consecutive gaps in the frame numbers
        groups = np.split(missing_t, np.where(np.diff(missing_t) != 1)[0] + 1)
        
        fake_detections = []
        toggle = True
        for g in groups:
            gap_length = len(g)
            
            # number of fake detections to insert
            n_fake = gap_length // insert_gap
            if n_fake > 0:
                # correctly space frames inside gap
                insert_frames = g[insert_gap-1::insert_gap]
        
                for frame in insert_frames:    
                    dummy_y, dummy_x = [-100, -100] if toggle else [-10, -10]
                    toggle = not toggle
                    fake = np.array( [dummy_y, dummy_x, frame, 0, -1, -1, -1, -1, -1, -1] ) 
        
                    fake_detections.append(fake)  
                    
        if len(fake_detections) > 0:
            fake_detections = np.array(fake_detections)
        
            # combine with original detections
            detections_filled = np.vstack((detections, fake_detections))
            
            # sort by frame
            detections_filled = detections_filled[np.argsort(detections_filled[:,2])]      
        else:
            detections_filled = detections.copy()
    
        return detections_filled



class trackastra_tracker(object):

    """
    tracking based on the tracking algorithm from: Trackastra: Transformer-based 
    cell tracking for live-cell microscopy, Gallusser, Benjamin and Weigert, Martin,
    ECCV 2024
    """
    
    def __init__(self, min_track_length=4, use_segmentations=True):
        super().__init__()

        from lift._helpers import _require_trackastra
        _require_trackastra()
        from trackastra.model import Trackastra
        import torch

        self.min_length = min_track_length
        self.use_segs = use_segmentations

        # Load a pretrained model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Trackastra.from_pretrained("general_2d", device=device)


    def track(self, tif_path):

        dir_path = Path(tif_path).parent
        img_stack = imread(tif_path)

        if self.use_segs:
            seg_path = dir_path / "spot_segmentation.tif"
            mask_stack = imread(seg_path)
        else: 
            detection_path = dir_path / "detected_foci.txt"  # get the path for the detections.txt
            detections = np.loadtxt(detection_path)
            mask_stack = self.coords_to_mask(detections, img_stack)

        # Track the spots
        track_graph, masks_tracked = self.model.track(img_stack, mask_stack, mode="greedy_nodiv") 
        tracks = self.track_graph_tolist(track_graph) 

        # filter out short tracks
        if self.min_length > 1:
            tracks = [trk for trk in tracks if len(trk) >= self.min_length]

        # save the resulting tracks
        out_xml =  dir_path / "tracks.xml"
        write_isbi_xml(tracks, out_xml)


    @staticmethod
    def track_graph_tolist(track_graph):
        """
        take the graphs of tracks generated by trackastra as input
        and output to a list like as tracks[track_idx][x, y, t]
        """
        import networkx as nx
        
        tracks = []
        for component in nx.weakly_connected_components(track_graph):

            subgraph = track_graph.subgraph(component)

            # extract nodes and sort by time
            nodes_sorted = sorted(subgraph.nodes(data=True), key=lambda x: x[1]['time'])

            track_array = []
            for node_id, node_attr in nodes_sorted:
              track_array.append([node_attr['coords'][1], node_attr['coords'][0], node_attr['time']] )

            tracks.append(np.array(track_array))

        return tracks

    @staticmethod
    def coords_to_mask(detections, image):
        mask_stack = np.zeros_like(image)
        for tt in range(mask_stack.shape[0]):
            detections_tt = detections[detections[:,2] == tt]
            detected_coords = detections_tt[:,:2].astype(int)

            labels = np.arange(len(detected_coords)) + 1  # preserve 0 for background

            if len(labels) > 0:
                xs, ys = detected_coords[:,0], detected_coords[:,1]
                mask_stack[tt, ys, xs] = labels
        
        return mask_stack