import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt, butter, savgol_filter
from scipy.ndimage import uniform_filter1d
configs=None
def buzsaki(lfp, fs, config=None, threshold=5, fall_off=2, sws_mask=None, light_mask=None, bandpass=(80, 250), lowpass=52.5, min_duration=15, clip_sd=5, verbose=False):
    """
    Detect ripples using the method described in Stark et al. (2014) / Roux et al. (2017) / Oliva et al. (2020).
    -> Stark threshold of 5 SD for detection, 2 SD for expansion, 15 ms min duration, 80-250 BP, LP = 52.5 Hz.
    -> Roux uses 5 SD., 2 SD., 20 ms min duration, 80-250 BP, LP = 52.5 Hz.
    -> Oliva et al. (2020) use 4 SD, 1 SD, 15 ms min duration, 100-300 BP, LP = 55 Hz.

    Algorithm:
    1. Band-pass filter (80–250 Hz).
    2. Compute instantaneous power:
       - Clip extreme values to 5 SD (of the band signal).
       - Rectify.
       - Low-pass filter (52.5 Hz).
    3. Determine Mean/SD from clipped power during SWS (non-theta, non-movement) & No Light.
    4. Compute power of original (unclipped) trace.
    5. Select events > 5 SD (from baseline mean). 
    6. Discard short events (<15 ms).
    7. Merge adjacent events (gap <15 ms).
    8. Expand events until power < 2 SD.
    9. Align by trough of band-pass signal closest to peak power.

    Parameters:
    -----------
    lfp : 1D array
        Raw LFP signal.
    fs : float
        Sampling frequency (Hz).
    config : str or dict, optional
        Configuration to use. Can be a string key (e.g. "stark2014") or a dictionary.
        Overrides individual parameters if provided.
    threshold : float
        Detection threshold (SD).
    fall_off : float
        Expansion threshold (SD).
    sws_mask : boolean array, optional
        Mask indicating Slow Wave Sleep (True=SWS). 
        If None, uses whole signal (not recommended for accurate baseline).
    light_mask : boolean array, optional
        Mask indicating Light Stimulation (True=Light ON).
        If None, assumes no light.
    bandpass : tuple
        (low, high) for bandpass filter.
    lowpass : float
        Cutoff for lowpass filter.
    min_duration : float
        Minimum duration (ms), also used for merge gap.
    verbose : bool
        Print info.

    Returns:
    --------
    events_df : pd.DataFrame
        Columns: [start_index, end_index, peak_index, peak_time, duration_ms, peak_zscore]
    """
    
    # Setup Config
    if config:
        if isinstance(config, str):
            if config in configs:
                cfg = configs[config]
            else:
                raise ValueError(f"Config '{config}' not found. Available: {list(configs.keys())}")
        elif isinstance(config, dict):
            cfg = config
        else:
            raise ValueError("Config must be a string (key) or a dictionary.")
        
        # Override defaults
        threshold = cfg.get("threshold", threshold)
        fall_off = cfg.get("fall_off", fall_off)
        bandpass = cfg.get("bandpass", bandpass)
        lowpass = cfg.get("lowpass", lowpass)
        min_duration = cfg.get("min_duration", min_duration)
        clip_sd = cfg.get("clip_sd", clip_sd)

    n_samples = len(lfp)
    
    if sws_mask is None:
        if verbose: print("Warning: No SWS mask provided. Using entire signal for baseline.")
        sws_mask = np.ones(n_samples, dtype=bool)
    
    if light_mask is None:
        light_mask = np.zeros(n_samples, dtype=bool)

    baseline_mask = sws_mask & (~light_mask)
    if np.sum(baseline_mask) == 0:
        if verbose: print("Warning: No clean baseline periods found (SWS + No Light). Using entire signal.")
        baseline_mask = np.ones(n_samples, dtype=bool)

    # 1. Band-pass filtering (80-250 Hz)
    # "zero-lag, linear phase FIR" -> firwin + filtfilt

    nyquist = fs / 2
    
    # 80 Hz -> 12.5 ms. 100ms window has 8 cycles. 
    numtaps = int(fs * 0.1) # 100ms
    if numtaps % 2 == 0: numtaps += 1
    
    b_band = firwin(numtaps, 
                    bandpass, 
                    pass_zero=False, 
                    fs=fs,
                    window='hann')   # smooth, Gaussian-like roll-off
    
    lfp_band = filtfilt(b_band, 1.0, lfp)

    # 2. Instantaneous Power Calculation (with Clipping)
    # Clip extreme values to 5 SD of the band signal (using baseline SD to act as ground truth for noise level)
    sd_band = np.std(lfp_band[baseline_mask])
    clip_thresh = clip_sd * sd_band
    
    lfp_band_clipped = np.clip(lfp_band, -clip_thresh, clip_thresh)
    lfp_rect = np.abs(lfp_band_clipped)
    
    # Low-pass filter (52.5 Hz, corresponding to pi cycles of mean band-pass)
    numtaps_lp = int(fs * 0.1) 
    if numtaps_lp % 2 == 0: numtaps_lp += 1
    
    b_lp = firwin(numtaps_lp, lowpass, pass_zero=True, fs=fs)
    power_clipped = filtfilt(b_lp, 1.0, lfp_rect)
    
    # 3. Baseline Statistics (Mean & SD of *Clipped* Power)
    power_baseline = power_clipped[baseline_mask]
    mean_base = np.mean(power_baseline)
    std_base = np.std(power_baseline)
    
    if verbose:
        print(f"Baseline (SWS+NoLight): Mean={mean_base:.2f}, SD={std_base:.2f}")

    # 4. Power of Original Trace (Unclipped)
    # "Subsequently, the power of the original trace was computed"
    # Implies: Rectify Unclipped Band -> Lowpass
    lfp_rect_orig = np.abs(lfp_band)
    power_orig = filtfilt(b_lp, 1.0, lfp_rect_orig)
    
    # 5. Event Selection
    # Thresholds
    thresh_detect = mean_base + threshold * std_base
    thresh_limit  = mean_base + fall_off * std_base # for expansion
    
    # Find islands > thresh_limit
    is_above_limit = power_orig > thresh_limit
    
    # Label connected components
    padded = np.concatenate(([0], is_above_limit, [0]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] # exclusive index
    
    candidates = []
    
    for s, e in zip(starts, ends):
        # Check if this segment contains any point > thresh_detect
        segment_power = power_orig[s:e]
        if np.max(segment_power) > thresh_detect:
            candidates.append((s, e))
            
    # 7. Merge adjacent events (gap < 15 ms)
    # 15 ms in samples
    # Using min_duration for gap as well, as per common practice in these papers if not specified otherwise
    gap_samples = int(min_duration / 1000 * fs)
    min_dur_samples = int(min_duration / 1000 * fs)
    
    merged_events = []
    if len(candidates) > 0:
        curr_start, curr_end = candidates[0]
        
        for next_start, next_end in candidates[1:]:
            if (next_start - curr_end) < gap_samples:
                # Merge
                curr_end = next_end
            else:
                merged_events.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_events.append((curr_start, curr_end))
    
    # 6. Discard short events (< 15ms)
    final_candidates = []
    for s, e in merged_events:
        if (e - s) >= min_dur_samples:
            final_candidates.append((s, e))
            
    # 9. Alignment by trough
    # "aligned by the trough (of the nonrectified signal) closest to the peak power"
    
    results = []
    if len(final_candidates) == 0:
        results.append({
            "start_s": np.nan,
            "end_s": np.nan,
            "trough_idx": np.nan,
            "trough_time": np.nan,
            "duration_ms": np.nan,
            "peak_zscore": np.nan,
            "thresh_cross_time": np.nan
        })
    for s, e in final_candidates:
        segment_power = power_orig[s:e]
        peak_offset = np.argmax(segment_power)
        peak_idx_power = s + peak_offset
        peak_val_power = segment_power[peak_offset]
        
        # Search for trough in lfp_band nearby the power peak
        search_radius = int(0.02 * fs) # 20ms radius
        search_s = max(s, peak_idx_power - search_radius)
        search_e = min(e, peak_idx_power + search_radius)
        
        # To find trough (minimum)
        seg_band = lfp_band[search_s:search_e]
        if len(seg_band) == 0:
            trough_idx = peak_idx_power
        else:
            trough_rel = np.argmin(seg_band)
            trough_idx = search_s + trough_rel
            
        zscore = (peak_val_power - mean_base) / std_base

        # Find first thresh_detect crossing
        cross_above = np.where(segment_power >= thresh_detect)[0]
        if len(cross_above) > 0:
            thresh_cross_idx = s + cross_above[0]
        else:
            thresh_cross_idx = s
    
        results.append({
            "start_s": s/fs,
            "end_s": e/fs,
            "trough_idx": trough_idx, # Aligned peak (actually trough)
            "trough_time": trough_idx / fs,
            "duration_ms": (e - s) * 1000 / fs,
            "peak_zscore": zscore,
            "thresh_cross_time": thresh_cross_idx / fs
        })
   
        
    return pd.DataFrame(results)

