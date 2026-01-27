import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt, butter, savgol_filter
from scipy.ndimage import uniform_filter1d

def buzsaki(lfp, fs, config=None, threshold=5, fall_off=2, sws_mask=None, light_mask=None, bandpass=(80, 250), lowpass=52.5, min_duration=15, clip_sd=5, verbose=False):
    """
    Detect ripples using the method described in Stark et al. (2014) / Roux et al. (2017) / Oliva et al. (2020).
    -> Stark threshold of 5 SD for detection, 2 SD for expansion, 15 ms min duration, 80-250 BP, LP = 52.5 Hz.
    -> Roux uses 2.5 SD., 2 SD., 20 ms min duration, 80-250 BP, LP = 52.5 Hz.
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
        
        results.append({
            "start_idx": s,
            "end_idx": e,
            "trough_idx": trough_idx, # Aligned peak (actually trough)
            "trough_time": trough_idx / fs,
            "duration_ms": (e - s) * 1000 / fs,
            "peak_zscore": zscore
        })
        
    return pd.DataFrame(results)

def cnn_liset(lfp, fs, config="cnn_liset", threshold=5.0, fall_off=1.5, bandpass=(100, 300), min_duration=15, verbose=False):
    """
    Implementation of the 'Gold Standard' filter-based detection used in CNN-Ripple (Navas-Olive et al., eLife, 2022).
    
    Reference Pipeline:
    1. Bandpass 2nd order Butterworth (100-300 Hz).
    2. Envelope: Rectified -> Savitzky-Golay (4th order) -> Smoothed (MovMean 2.3ms and 6.7ms).
    3. Thresholding:
       - First Threshold (Edge): 'fall_off' * SD (range 1-2.5 in paper).
       - Second Threshold (Detection): 'threshold' * SD (range 3-10 in paper).
    4. Merge: gaps < 15ms.

    Parameters:
    -----------
    lfp : array_like
        LFP signal.
    fs : float
        Sampling frequency.
    config : str/dict, optional
        Predefined config name (e.g., 'cnn_liset') or dict.
    threshold : float
        Second/High threshold for detection (SD).
    fall_off : float
        First/Low threshold for event start/end (SD).
    bandpass : tuple
        (low, high). Default (100, 300).
    min_duration : float
        Minimum duration and merge gap (ms). Default 15.
        
    Returns:
    --------
    pd.DataFrame containing event info.
    
    Thresholds ->
    To look for the fairest comparison, we made predictions for all possible combinations:
    Low threshold: 1, 1.5, 2, 2.5 SD.
    High threshold: 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10 SD 
    We then chose the one that scored the maximum F1. This was done separately for each session.
    """
    
    # 1. Config override
    if config:
        if isinstance(config, str):
            cfg = configs.get(config, {})
        elif isinstance(config, dict):
            cfg = config
        else:
            cfg = {}
        
        threshold = cfg.get("threshold", threshold)
        fall_off = cfg.get("fall_off", fall_off)
        bandpass = cfg.get("bandpass", bandpass)
        min_duration = cfg.get("min_duration", min_duration)

    # 2. Bandpass Filter (Butterworth 2nd order [effectively 4th via filtfilt], 100-300Hz)
    b, a = butter(2, bandpass, btype='bandpass', fs=fs)
    filtered = filtfilt(b, a, lfp)

    # 3. Envelope Calculation
    # "Amplified twice"-> multiply by 4?
    amplified = filtered*4 #TODO: This does not make sense, and is left ambiguous in the paper. Check with authors if needed.
    
    # Savitzky-Golay (4th order). Small frame 
    # Example: 1ms ~ 1.25 samples at 1250Hz. If FS is high (30k), 33 samples is ~1ms.
    # We will use a minimum window of 5 or approx 3-4ms equivalent if feasible.
    sg_window = max(5, int(fs * 0.005)) # ~5ms or 5 samples min
    if sg_window % 2 == 0: sg_window += 1
    
    envelope = savgol_filter(amplified, window_length=sg_window, polyorder=4)
    
    # Movmean smoothing (2.3 ms and 6.7 ms)
    w1 = max(1, int(2.3 * fs / 1000))
    if w1 > 1:
        envelope = uniform_filter1d(envelope, size=w1, mode='constant', cval=0)
        
    w2 = max(1, int(6.7 * fs / 1000))
    if w2 > 1:
        envelope = uniform_filter1d(envelope, size=w2, mode='constant', cval=0)

    # 4. Thresholding
    # SD calculated over the whole session/signal
    env_mean = np.mean(envelope)
    env_std = np.std(envelope)
    
    thresh_high = env_mean + threshold * env_std
    thresh_low = env_mean + fall_off * env_std
    
    # Logic: Find segments > Low. Keep if they cross High.
    is_above_low = envelope > thresh_low
    
    # Connected components
    padded = np.concatenate(([0], is_above_low, [0]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    candidates = []
    
    # Merge gap logic (15ms)
    gap_samples = int(min_duration / 1000 * fs)
    
    # Pre-merge list of valid candidates
    valid_intervals = []
    for s, e in zip(starts, ends):
        seg = envelope[s:e]
        if np.max(seg) >= thresh_high:
            valid_intervals.append((s, e))
            
    # Merge Phase
    merged_events = []
    if len(valid_intervals) > 0:
        curr_s, curr_e = valid_intervals[0]
        
        for next_s, next_e in valid_intervals[1:]:
            if (next_s - curr_e) < gap_samples:
                curr_e = next_e # Extend
                # Note: If we merge two events, the combined event definitely crosses high thresh
            else:
                merged_events.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        merged_events.append((curr_s, curr_e))
        
    results = []
    for s, e in merged_events:
        dur_samples = e - s
        if dur_samples < 1: continue 
        
        seg = envelope[s:e]
        peak_offset = np.argmax(seg)
        peak_idx = s + peak_offset
        peak_val_env = seg[peak_offset]
        zscore = (peak_val_env - env_mean) / env_std
        
        # Trough alignment (similar to other methods, though prompt doesn't explicitly demand trough alignment for detection, usually required for metric)
        # We can just return peak of envelope time
        
        results.append({
            "start_idx": s,
            "end_idx": e,
            "peak_index": peak_idx, 
            "peak_time": peak_idx / fs,
            "duration_ms": (dur_samples / fs) * 1000,
            "peak_zscore": zscore
        })
        
    return pd.DataFrame(results)

def get_ripple_events(lfp,fs,config="stark2014",**kwargs):
    """
    Wrapper to get ripple events using specified config.
    """
    if config in configs:
        func = configs[config]["function"]
    else:
        raise ValueError(f"Config '{config}' not found. Available: {list(configs.keys())}")
    
    return func(lfp, fs, config=config, **kwargs)




# Predefined Configurations for recreating different papers.
configs={
    "stark2014":{
        "function":buzsaki,
        "threshold":5,
        "fall_off":2,
        "bandpass":(80,250),
        "lowpass":52.5,
        "min_duration":15,
        "clip_sd":5
    },
    "roux2017":{
        "function":buzsaki,
        "threshold":2.5,
        "fall_off":2,
        "bandpass":(80,250),
        "lowpass":52.5,
        "min_duration":15,
        "clip_sd":5},
    "oliva2020":{
        "function":buzsaki,
        "threshold":4,
        "fall_off":1,
        "bandpass":(100,300),
        "lowpass":55,
        "min_duration":15,
        "clip_sd":4},
    "cnnliset":{
        "function":cnn_liset,
        "threshold": 5.0,  # Detection threshold (Second threshold in text)
        "fall_off": 1.0,   # Start/End threshold (First threshold in text)
        "bandpass": (100, 300),
        "min_duration": 15, # Merging gap and min duration are often same or similar
    }
}