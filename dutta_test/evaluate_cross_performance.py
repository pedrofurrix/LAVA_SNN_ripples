import numpy as np
from load_gt_crossval import *
import itertools
import pandas as pd

def compute_metrics_single_channel(
    spikes_ms, 
    ripples_ms, 
    tolerance=20, 
    max_detection_offset=100, 
    fp_grouping_window=50, # jitter
    extra_tolerance=50
):
    """
    Compute TP, FP, FN for a single channel.
    
    Args:
        spikes_ms: 1D array of spike times in ms.
        ripples_ms: 2D array of ripple intervals [start, end] in ms.
        tolerance: Tolerance in ms (added before ripple start).
        max_detection_offset: Max duration in ms to look for a spike after ripple start.
        fp_grouping_window: Window in ms to group consecutive FP spikes into a single FP event.
    """
    
    if len(ripples_ms) == 0:
        # No ripples: all spikes are FPs
        # Group FPs
        fp_count = 0
        if len(spikes_ms) > 0:
            sorted_spikes = np.sort(spikes_ms)
            current_fp_end = -np.inf
            for spk in sorted_spikes:
                if spk > current_fp_end:
                    fp_count += 1
                    current_fp_end = spk + fp_grouping_window # jitter 
                else:
                    # Extend window? Or just ignore? 
                    # Usually we just count events. Let's say the event lasts fp_grouping_window.
                    pass
        return 0, fp_count, 0, [] # TP, FP, FN, latencies

    # Define valid detection windows for each ripple
    # Window: [start - tolerance, start + max_detection_offset + tolerance]
    # Note: max_detection_offset is usually relative to start.
    if ripples_ms.ndim==1:
        ripple_starts = ripples_ms
    else:
        ripple_starts = ripples_ms[:, 0]
    valid_windows = []
    for start in ripple_starts:
        w_start = start - tolerance
        w_end = start + max_detection_offset
        valid_windows.append((w_start, w_end))
    
    # Sort spikes
    spikes_ms = np.sort(spikes_ms)
    
    tp_count = 0
    fn_count = 0
    latencies = []
    
    # Track which spikes are used for TPs to exclude them from FP count
    used_spike_indices = set()
    
    # 1. Check for TPs and FNs
    for r_idx, (w_start, w_end) in enumerate(valid_windows):
        # Find spikes in window
        # Search sorted
        idx_start = np.searchsorted(spikes_ms, w_start) # find first spike >= w_start
        idx_end = np.searchsorted(spikes_ms, w_end) # find first spike > w_end

        in_window_indices = np.arange(idx_start, idx_end) # if not same, there are spikes in window
        
        if len(in_window_indices) > 0:
            tp_count += 1
            # Calculate latency (first spike relative to ripple start)
            first_spike = spikes_ms[in_window_indices[0]]
            latencies.append(first_spike - ripple_starts[r_idx])
            
            # Mark these spikes as part of a TP (so not FPs)
            # Note: A single spike could theoretically trigger multiple ripples if they overlap heavily,
            # but usually we just mark it used.
            for idx in in_window_indices:
                used_spike_indices.add(idx)
        else:
            fn_count += 1
            
    # 2. Count FPs (grouping remaining spikes)
    fp_count = 0
    current_fp_end = -np.inf
    
    for i, spk in enumerate(spikes_ms):
        if i in used_spike_indices:
            continue
            
        # Check if this spike falls into ANY valid ripple window (even if that ripple was already detected by another spike)
        # This prevents counting extra spikes during a detected ripple as FPs.
        # Optimization: We could merge all valid windows, but simple check is okay for now.
        is_in_valid_window = False

        # add extra tolerance to avoid edge cases
        valid_windows_extended = [(w_start, w_end + extra_tolerance) for (w_start, w_end) in valid_windows]
        for w_start, w_end in valid_windows_extended:
            if w_start <= spk <= w_end:
                is_in_valid_window = True
                break
        
        if is_in_valid_window:
            continue
            
        # It's an FP candidate
        if spk > current_fp_end:
            fp_count += 1
            current_fp_end = spk + fp_grouping_window
            
    return tp_count, fp_count, fn_count, latencies


def process_cross_val_results(
    gt_id,
    detect_id,
    Loaders,
    tolerance=20,
    max_detection_offset=100,
    fp_grouping_window=50,
    extra_tolerance=50,
    output_csv=None,
    network_name=None,
    threshold=None,
    tolerance_gt=100,
):
    results=[]
    GT_loader=Loaders[gt_id](tolerance=tolerance_gt)
    Detections_loader=Loaders[detect_id]()
    gt_ms=GT_loader.GT
    detections_ms=Detections_loader.detections

    for session in gt_ms.keys():
        if session not in detections_ms:
            detections_ms[session]=np.array([])
        tp, fp, fn, latencies = compute_metrics_single_channel(
            detections_ms[session], 
            gt_ms[session], 
            tolerance=tolerance, 
            max_detection_offset=max_detection_offset,
            fp_grouping_window=fp_grouping_window,
            extra_tolerance=extra_tolerance
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_latency = np.mean(latencies) if latencies else np.nan

        print(f"  TP: {tp}, FP: {fp}, FN: {fn} -> F1: {f1:.4f}")
        
        results.append({
            "GT": gt_id,
            "Detections": detect_id, 
            "Session": session,
            "Threshold": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Mean_Latency": mean_latency,
            "Num_Spikes": len(detections_ms[session]),
            "Num_Ripples": len(gt_ms[session])
        })
    return results


Loaders={
"SNN_GT":SNN_GT,
"RippleNet_GT":RippleNet_GT,
"RipplAI_GT":RipplAI_GT,
"LisetCNN_GT":LisetCNN_GT,
"Dutta_GT":Dutta_GT
}
 
if __name__=="__main__":
    combinations=itertools.permutations(Loaders.keys(),2)
    all_results=[]
    for combination in combinations:
        gt_id, detect_id = combination
        print(f"Evaluating GT: {gt_id} vs Detections: {detect_id}")
        results=process_cross_val_results(
            gt_id,
            detect_id,
            Loaders,
            tolerance=60,
            max_detection_offset=60,
            fp_grouping_window=100,
            extra_tolerance=100,
            output_csv=None,
            network_name=None,
            threshold=None,
            tolerance_gt=100,
        )
        all_results.extend(results)
    data_path=os.path.join(os.path.dirname(__file__),"spikes","crossval_results.csv")
    df=pd.DataFrame(all_results)
    df.to_csv(data_path,index=False)
    print(f"Saved cross-validation results to {data_path}")
