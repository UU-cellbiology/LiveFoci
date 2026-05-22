from pathlib import Path
import os
import shutil
from xml.dom import minidom 
import subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment
import skimage
from lift.general_utils import load_sequence



#####
#
# below is the helper functions to run the different tracking algorithms on a folder and save the resulting output
#
#####


def run_tracker(path_list, method=None, min_length=20, **kwargs):
    """
    performs nuclei tracking using the specified tracking method 
    
    path_list: should be a list of paths with the folders of segmented masks that you would like to track. 
    method: string specifying which tracking method to use. this tracking method takes the segmented masks
    as input to perform the linking between the nuclei over time. The method should save the output according
    to the format of the cell tracking challange
    min_length: the minimum allowed length of the tracks in the number of frames

    optional parameters:    
    iou_min: IOU tracker only, the minimum overlap required before considering it to be linked
    gap_closing: NND tracker only, number of frames to close between tracks when detections are missing
    max_distance: NND tracker only, the maximum allowed distance for linking detections [pixels]
    remove_gaps: for Trackastra only, when True splits tracks into separate gaps whenever there is 
        a missing detection
    """
    if method is None:
        from lift._helpers import _detect_nuclei_tracker
        method = _detect_nuclei_tracker()

    print(f"── nuclei tracking method:   {method}")

    if method == 'IOU':
        iou_min = kwargs.pop("iou_min", 0.01)
        tracker = IOU_tracker(min_track_length=min_length, iou_min=iou_min)
    elif method == 'NND':
        max_dist  = kwargs.pop("max_distance", 30.0)
        gap_close = kwargs.pop("gap_closing",  0)
        tracker   = NND_tracker(min_track_length=min_length, max_distance=max_dist, gap_closing=gap_close)
    elif method == 'trackastra':
        remove_gaps = kwargs.pop("remove_gaps", True)
        tracker     = trackastra_tracker(min_track_length=min_length, remove_gaps=remove_gaps)
    else:
        raise ValueError(f"Unknown nuclei tracking method: {method!r}")

    for path in path_list:
        path = Path(path)
        tracker.track(path)


#####
#
# Below are the classes for the different tracking algorithms
#
#####



class IOU_tracker(object):
    
    def __init__(self, min_track_length=20, iou_min=0.01):
        super().__init__()

        self.iou_min = iou_min, 
        self.min_track_length = min_track_length


    def track(self, path):

        path = Path(path)
        segmentation_sequence = load_sequence.load(path)

        tracked_sequence, track_info = self.iou_assignment(segmentation_sequence)            
        
        save_path = path.parent / "cell_tracking"
        save_path.mkdir(exist_ok=True)
        self.save_to_ctc_format(tracked_sequence, track_info, save_path)
        

    def iou_assignment(self, segmentation_sequence):
    
        track_info = {}
        tracked_sequence = []
        
        next_track_id = 1
        
        prev_labels = None
        prev_track_ids = None
        prev_seg = None
        
        for t, seg in enumerate(segmentation_sequence):
            out = np.zeros_like(seg)
            curr_labels = np.unique(seg)[1:]  # remove zero as it is the background label
        
            if t == 0:
                # initialize tracks
                curr_track_ids = []
                for l in curr_labels:
                    out[seg == l] = next_track_id
                    curr_track_ids.append(next_track_id)
                    track_info[next_track_id] = {"t0": t, "t1": t, "parent": 0}
                    
                    next_track_id += 1
            else:
                n_prev = len(prev_labels)
                n_curr = len(curr_labels)
        
                iou_mat = np.zeros((n_prev, n_curr))
                label_to_index = {label: j for j, label in enumerate(curr_labels)}
                for i, l_prev in enumerate(prev_labels):
                    mask_prev = prev_seg == l_prev
                    
                    # filter candidates to not compute IOU for cells that not overlap at all
                    candidate_labels = np.unique(seg[mask_prev])
                    candidate_labels = candidate_labels[candidate_labels != 0] # remove background as candidate
                    for L_curr in candidate_labels:
                        j = label_to_index[L_curr]
                        mask_curr = (seg == L_curr)
                        iou_mat[i, j] = self.iou(mask_prev, mask_curr)
        
                cost = 1 - iou_mat
                row_ind, col_ind = linear_sum_assignment(cost)
        
                curr_track_ids = [None] * n_curr
                idx_assigned_curr = set()
        
                # propagate existing tracks
                for i, j in zip(row_ind, col_ind):
                    if iou_mat[i, j] >= self.iou_min:
                        track_id = prev_track_ids[i]
                        curr_track_ids[j] = track_id
                        out[seg == curr_labels[j]] = track_id
                        idx_assigned_curr.add(j)
                        track_info[track_id]["t1"] = t
        
                # start new tracks
                for j in range(n_curr):
                    if j not in idx_assigned_curr:
                        curr_track_ids[j] = next_track_id
                        out[seg == curr_labels[j]] = next_track_id
                        track_info[next_track_id] = {"t0": t, "t1": t, "parent": 0}
                        next_track_id += 1
        
        
            tracked_sequence.append(out)
        
            # advance state
            prev_labels = curr_labels
            prev_track_ids = curr_track_ids
            prev_seg = seg.copy()    
        
        if self.min_track_length > 1:
            tracked_sequence, track_info = self.filter_short_tracks(tracked_sequence, track_info)
   
        return tracked_sequence, track_info

    @staticmethod
    def iou(mask1, mask2):
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union > 0 else 0
    

    def filter_short_tracks(self, tracked_sequence, track_info):
        keep_tracks = set()
    
        for track_id, info in track_info.items():
            length = info["t1"] - info["t0"] + 1
            if length >= self.min_track_length:
                keep_tracks.add(track_id)

        pruned_sequence = self.prune_sequence(tracked_sequence, keep_tracks)
        pruned_info = self.prune_track_info(track_info, keep_tracks)
        pruned_sequence, pruned_info = self.relabel_tracks_consecutively(pruned_sequence, pruned_info)
    
        return pruned_sequence, pruned_info

    
    @staticmethod
    def prune_sequence(tracked_sequence, keep_tracks):
        keep_tracks = np.array(list(keep_tracks))
        filtered_sequence = []
    
        for frame in tracked_sequence:
            mask = np.isin(frame, keep_tracks)
            out = frame * mask
            filtered_sequence.append(out)
    
        return filtered_sequence
    

    @staticmethod
    def prune_track_info(track_info, keep_tracks):
        return {
            track_id: info
            for track_id, info in track_info.items()
            if track_id in keep_tracks
        }

    
    @staticmethod
    def relabel_tracks_consecutively(tracked_sequence, track_info):
        new_ids = {tid: i+1 for i, tid in enumerate(sorted(track_info))} # map old id to new id
        new_sequence = []
    
        for frame in tracked_sequence:
            out = np.zeros_like(frame)
            for old_id, new_id in new_ids.items():
                out[frame == old_id] = new_id
            new_sequence.append(out)
    
        new_track_info = {}
        for old_id, info in track_info.items():
            new_track_info[new_ids[old_id]] = info
    
        return new_sequence, new_track_info

    
    @staticmethod
    def track_info_to_array(track_info):
        """
        Convert track_info dict to a NumPy array suitable for np.savetxt
        """
        rows = []
    
        for track_id in sorted(track_info.keys()):
            info = track_info[track_id]
            rows.append([track_id, info['t0'], info['t1'], info['parent']])
    
        return np.array(rows, dtype=int)
    

    def save_to_ctc_format(self, tracked_sequence, tracking_info, save_path):
        # Save t0000.tif, t0001.tif, ...
        for tt, tracked in enumerate(tracked_sequence):
            save_file = save_path / f"t{tt:04}.tif"
            skimage.io.imsave(str(save_file), tracked.astype(np.uint32), check_contrast=False)

        # Save tracking file
        track_file = save_path / "res_track.txt"
        np.savetxt(str(track_file), self.track_info_to_array(tracking_info), fmt="%d")


class NND_tracker(object):
    
    def __init__(self, min_track_length, max_distance, gap_closing=0, motion_model=0):
        super().__init__() 

        self.mm = motion_model
        
        #get directory for this class
        self.base_dir =  Path(__file__).parent.resolve()  #Path.cwd()  #

        if os.name == "nt":  # command for windows
            self.java = self.base_dir / "tracker_utils" / "ImageJ" / "jre" / "bin" / "java.exe"
            self.classpath_sep = ";"
        else:  # command for other systems
            self.java = self.base_dir / "tracker_utils" / "ImageJ" / "jre" / "bin" / "java"
            self.classpath_sep = ":"
                            
        # location of the java tracking plugin
        self.TP_dir = self.base_dir / "tracker_utils" / "SOSTracker commandline"

        # Classpath jars
        jars = ["VENI_.jar", "ij.jar", "imagescience.jar", "Jama-1.0.2.jar"]
        classpath = self.classpath_sep.join(jars)

        self.plugins = ["-Xmx3000m", "-cp", classpath, "ws.smal.sos.eval.linking.Linker"]

        # modify the parameter file to use the user specified parameters        
        self.modify_parameter_file(min_track_length, max_distance, gap_closing)
                
        
    def track(self, path, tif_file=None):
        
        seg_path = Path(path)

        # make the output folder
        save_path = seg_path.parent / "cell_tracking" 
        save_path.mkdir(exist_ok=True)

        # make a folder to store the files needed for the tracker
        temp_path = save_path / "temp_files"
        temp_path.mkdir(exist_ok=True)

        #convert the segmentation to detections.xml file
        seg_sequence = self.convert_to_detections(seg_path, temp_path)
        num_frames = len(seg_sequence)
        
        if not tif_file:
            # create a fake file called .tif as required for tracking plugin'
            tif_file = temp_path / ".tif"
            shutil.copy(temp_path / "detections.xml.txt", tif_file)

        # copy parameters.xml file to folder of the image
        shutil.copy(self.TP_dir / "modified_parameters.xml", temp_path / "parameters.xml")

        args = [str(tif_file), str(num_frames), str(self.mm)]

        # final command for the tracking
        cmd = [str(self.java), *self.plugins, *args]

        p = subprocess.run(cmd, capture_output=True, text=True, cwd=self.TP_dir)
        #print(p.stderr)
        #print(p.stdout)   
        
        # save the results to the format of the cell tracking challange
        self.save_to_ctc_format(save_path, temp_path, seg_sequence)


    def modify_parameter_file(self, min_track_length, max_distance, gap_closing):    
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

            if param_id == "nrOfGapsToClose":
                param.setAttribute("value", f"{gap_closing:.1f}")
        
        # Save to new file
        output_file = self.TP_dir / "modified_parameters.xml"
        with open(output_file, "w") as f:
            doc.writexml(f, indent="  ", addindent="  ", newl="\n")

    @staticmethod
    def save_to_ctc_format(save_path, track_path, segmentations):
            # convert the tracking back in the format of the cell tracking challenge
            tracking_file = str(track_path / f"{track_path.name}.NND.xml")
            detections = str(track_path / "detections.xml.txt")
            
            tree = minidom.parse(tracking_file)
            tracks = tree.getElementsByTagName('particle')
            detections = np.loadtxt(detections)
            tracked_labels = [np.zeros(segmentations[0].shape, dtype=np.uint32) for _ in segmentations]
            label_counter = 0
            for track in tracks:   
                tracked_cell = track.getElementsByTagName('detection')
                label_counter += 1
                for trck_cell in tracked_cell:
                    t = int(trck_cell.attributes['t'].value)
                    x = float(trck_cell.attributes['x'].value)
                    y = float(trck_cell.attributes['y'].value)
            
                    idx = np.where((detections[:, 0] == x) & 
                                   (detections[:, 1] == y) & 
                                   (detections[:, 2] == t))[0][0]
                    old_label = detections[idx,4]
            
                    tracked_labels[t][segmentations[t] == old_label] = label_counter
            
            # save the image with unique track labels
            for tt, track_lab in enumerate(tracked_labels):
                save_file = save_path / f"t{tt:04}.tif"
                skimage.io.imsave(str(save_file), tracked_labels[tt], check_contrast=False)
            
            # write the corresponding text file
            labels = np.unique(np.array(tracked_labels))[1:]  # 0 is the background 
            output_file = []
            for ll in labels:        
                indices = np.where(np.any(np.array(tracked_labels) == ll, axis=(1, 2)))[0]
                t1 = indices[0]
                t2 = indices[-1]
                
                parent = 0  # we don't track divisions so no parents
                
                output_file.append([ll, t1, t2, parent])
            
            output_file = np.vstack(output_file)
            np.savetxt(str(save_path / "res_track.txt"), output_file, fmt="%d")

    
    @staticmethod        
    def convert_to_detections(in_fold, out_folder):
        
        segmentation_stack = load_sequence.load(in_fold)
           
        all_detections = []
        for tt, segmentation in enumerate(segmentation_stack):        
                        
            regions = skimage.measure.regionprops(segmentation)
            
            for cell in regions:
                x, y = cell.centroid
                cell_area = cell.area
                label = cell.label
                    
                temp = np.array( [y, x, tt, cell_area, label, 0.0, 0.0, 0.0, 0.0, 0.0] ).T                   
                all_detections.append(temp)
            
        all_detections = np.vstack(all_detections) 
                
        np.savetxt(out_folder / "detections.xml.txt", all_detections, fmt='%.2f', delimiter='\t')
        
        return segmentation_stack



class trackastra_tracker(object):

    """
    tracking based on the tracking algorithm from: Trackastra: Transformer-based 
    cell tracking for live-cell microscopy, Gallusser, Benjamin and Weigert, Martin,
    ECCV 2024
    """
    
    def __init__(self, min_track_length=4, remove_gaps=True):
        super().__init__()
        from lift._helpers import _require_trackastra
        _require_trackastra()
        from trackastra.model import Trackastra
        from trackastra.tracking import graph_to_ctc
        import torch

        self.min_length = min_track_length
        self.remove_gaps = remove_gaps

        # Load a pretrained model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Trackastra.from_pretrained("ctc", device=device)
        self.track_saver = graph_to_ctc

    
    def track(self, seg_path):

        seg_path = Path(seg_path)
        mask_stack = np.stack(load_sequence.load(seg_path))

        img_path = seg_path.parent.parent / "raw"
        img_stack = np.stack(load_sequence.load(img_path))        

        # Track the nuclei
        track_graph, masks_tracked = self.model.track(img_stack, mask_stack, mode="greedy_nodiv") 

        # split tracks on gaps and relabel masks
        if self.remove_gaps:
            track_graph = self.split_tracks_on_gaps(track_graph)

        # remove short tracks
        if self.min_length > 1:
            track_graph, masks_tracked = self.filter_short_tracks(track_graph, masks_tracked)

        # write the result to ctc format
        save_path = str(seg_path.parent / "cell_tracking")
        ctc_tracks, ctc_masks = self.track_saver(track_graph, masks_tracked, outdir=save_path)
        self.rename_ctc_outputs(save_path, n_frames=mask_stack.shape[0])


    def split_tracks_on_gaps(self, track_graph):
        """
        Split tracks when temporal gaps occur by removing edges.
        Masks are NOT modified because graph_to_ctc relies on the
        original segmentation labels.
        """
    
        edges_to_remove = []   
        for u, v in track_graph.edges():    
            t1 = track_graph.nodes[u]["time"]
            t2 = track_graph.nodes[v]["time"]
    
            if t2 - t1 > 1:
                edges_to_remove.append((u, v))
    
        track_graph.remove_edges_from(edges_to_remove)
    
        return track_graph


    def filter_short_tracks(self, track_graph, masks_tracked):
    
        import networkx as nx
    
        nodes_to_remove = []    
        for component in nx.weakly_connected_components(track_graph):
    
            subgraph = track_graph.subgraph(component)
    
            if len(subgraph.nodes) < self.min_length:
                nodes_to_remove.extend(list(subgraph.nodes))
    
        for node in nodes_to_remove:    
            attr = track_graph.nodes[node]
    
            t = attr["time"]
            label = attr["label"]    
            masks_tracked[t][masks_tracked[t] == label] = 0
    
        track_graph.remove_nodes_from(nodes_to_remove)
    
        return track_graph, masks_tracked

    
    @staticmethod
    def rename_ctc_outputs(outdir, n_frames):
        """
        rename the outputs for compatibility with ctc quantification and consistency with other trackers
        """
        
        outdir = Path(outdir)
    
        # rename track file
        man_track = outdir / "man_track.txt"
        res_track = outdir / "res_track.txt"
        os.replace(man_track, res_track)
    
        # rename mask files
        for t in range(n_frames):        
            old_name = outdir / f"man_track{t:04d}.tif"
            new_name = outdir / f"t{t:04d}.tif"
            os.replace(old_name, new_name)
            