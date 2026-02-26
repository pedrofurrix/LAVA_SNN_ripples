import os
import pickle
import numpy as np
import sys
import pandas as pd
from scipy.signal import hilbert
from tqdm import tqdm
import re
from scipy.signal import periodogram
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# Add path to project root
curr_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR= os.path.abspath(os.path.join(curr_dir, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from liset_data_reader.read_data import read_data
from liset_data_reader.liset_tk_extra import liset_tk_extra
import liset_data_reader.lists_sessions as lists_sessions
from liset_data_reader.ripple_band_power import ripple_band_power_trace
from snnTorch.generalization_madrid.process_signal import load_experimental_data,spikify_signal
from liset_data_reader.signal_aid import bandpass_filter

def exponential_func(x, a, b):
    """ Exponential curve: a * exp(b * x) """
    return a * np.exp(b * x)

def calculate_preferred_frequency(signal, fs, band=(70, 400), min_resolution=10.0):
    """
    Computes the PSD of a 100ms window, fits an exponential curve (A⋅e^(B⋅f)) to the 70-400 Hz band, 
    subtracts it, and finds the peak.
    
    Args:
        min_resolution (float): Desired frequency resolution in Hz. 
                                Uses zero-padding (nfft) to achieve this.
    """
    # Calculate required nfft to achieve desired resolution: df = fs / nfft
    n_samples = len(signal)
    nfft = max(n_samples, int(fs / min_resolution))
    
    f, Pxx = periodogram(signal, fs, scaling='density', nfft=nfft)
     # f - array of sample frequencies:
     # Pxx - Power spectral density or power spectrum of x.

    # Select Band
    mask = (f >= band[0]) & (f <= band[1])
    f_sub = f[mask]
    P_sub = Pxx[mask]
    
    if len(f_sub) < 4:
        return np.nan
        
    # Fit Exponential
    try:
        p0 = [np.max(P_sub), -0.005]
        popt, _ = curve_fit(exponential_func, f_sub, P_sub, p0=p0, maxfev=5000)
        #curve_fit(f, xdata, ydata, p0=None,)
    except:
        return f_sub[np.argmax(P_sub)] # Fallback to max frequency power
        
    if popt is None:
        return f_sub[np.argmax(P_sub)]

    P_fit = exponential_func(f_sub, *popt)
    P_corr = P_sub - P_fit
    
    return f_sub[np.argmax(P_corr)] # NICE # CHECKED

def calculate_low_freq_contribution(signal, fs, cutoff=100):
    signal = signal - np.mean(signal)

    f, Pxx = periodogram(signal, fs, scaling='density')

    total_power = np.sum(Pxx)  # FULL spectrum

    if total_power == 0:
        return 0

    low_power = np.sum(Pxx[f < cutoff])

    return low_power / total_power

def calculate_spectral_entropy(signal, fs):
    """
    Computes Shannon entropy of normalized power spectrum.
    """
    f, Pxx = periodogram(signal, fs, scaling='density')
    
    # Select Band
    mask = (f >= 70) & (f <= 400)
    f_sub = f[mask]
    P_sub = Pxx[mask]

    total_power = np.sum(P_sub)
    if total_power == 0:
        return np.nan
        
    P_norm = P_sub / total_power
    
    # Filter zeros for log
    mask = P_norm > 0
    P_nz = P_norm[mask]
    
    return -np.sum(P_nz * np.log2(P_nz))

def fit_exponential(f, P):

    """
    Fits P = a * exp(b * f).
    Returns parameters [a, b] or None if fit fails.
    """
    try:
        # We'll use standard least squares.
        p0 = [np.max(P), -0.005]
        # bounds = ([0, -np.inf], [np.inf, 0])
        popt, _ = curve_fit(exponential_func, f, P, p0=p0, maxfev=5000)
        return popt
    except Exception as e:
        # print(f"Fit failed: {e}")
        return None
    
def extract_events(spikes_ms, ripples_samp, fs=30000, tolerance_ms=20, max_offset_ms=100,extra_tolerance_ms=20, fp_grouping_ms=50):
    """
    Classify events into TP, FP, FN and return their details.
    """
    # Convert spikes to samples
    spikes_samp = (spikes_ms * (fs / 1000)).astype(int)
    
    # Tolerance in samples
    tol_samp = int(tolerance_ms * fs / 1000)
    max_offset_samp = int(max_offset_ms * fs / 1000)
    fp_group_samp = int(fp_grouping_ms * fs / 1000)
    extra_tolerance_samp= int(extra_tolerance_ms * fs / 1000)
    # Ripple windows for detection
    ripple_windows = []
    for r in ripples_samp:
        start = int(r[0])
        end = int(r[1])
        # Detection window: [start - tol, start + max_offset]
        w_start = start - tol_samp
        w_end = start + max_offset_samp
        ripple_windows.append((w_start, w_end, start, end))
        
    tp_events = [] 
    fp_events = [] 
    fn_events = [] 
    
    detected_ripple_indices = set()
    used_spike_indices = set()
    
    # Sort spikes
    sorted_spike_indices = np.argsort(spikes_samp)
    sorted_spikes = spikes_samp[sorted_spike_indices]
    
    # 1. Find TPs and FNs
    for r_idx, (w_start, w_end, r_start, r_end) in enumerate(ripple_windows):
        
        # Find spikes in window
        idx_start = np.searchsorted(sorted_spikes, w_start)
        idx_end = np.searchsorted(sorted_spikes, w_end)
        
        in_window_indices = np.arange(idx_start, idx_end)
        
        if len(in_window_indices) > 0:
            # TP
            detected_ripple_indices.add(r_idx)
            
            # First spike is the detection
            first_spike_idx_in_sorted = in_window_indices[0]
            first_spike_samp = sorted_spikes[first_spike_idx_in_sorted]
            
            tp_events.append({
                'spike_samp': first_spike_samp,
                'ripple_idx': r_idx,
                'ripple_start': r_start,
                'ripple_end': r_end
            })
            
            # Mark all spikes in window as used
            for idx in in_window_indices:
                used_spike_indices.add(idx)
        else:
            # FN
            fn_events.append({
                'ripple_idx': r_idx,
                'ripple_start': r_start,
                'ripple_end': r_end
            })
            
    # 2. Find FPs
    current_fp_end = -np.inf
    
    for i in range(len(sorted_spikes)):
        if i in used_spike_indices:
            continue
            
        spk = sorted_spikes[i]
        
        # Check if this spike falls into ANY ripple window (even if that ripple was already detected)
        # This prevents counting extra spikes during a detected ripple as FPs
        is_in_ripple = False
        for w_start, w_end, _, _ in ripple_windows:
             # Using the same detection window logic for exclusion
             if w_start <= spk <= w_end + extra_tolerance_samp:
                 is_in_ripple = True
                 break
        
        if is_in_ripple:
            continue
            
        if spk > current_fp_end:
            # Find the last spike within the grouping window to define event boundaries
            limit_idx = np.searchsorted(sorted_spikes, spk + fp_group_samp, side='right')
            last_spk_in_group = sorted_spikes[limit_idx - 1]

            fp_events.append({
            'spike_samp': spk,
            'ripple_start': spk,
            'ripple_end': last_spk_in_group
            })
            current_fp_end = spk + fp_group_samp
            
    return tp_events, fp_events, fn_events

def extract_power_snippets(power_trace, events, event_type, window_samp, look_around_ms=50, FS=30000,center="peak"):
    snippets = []
    max_idx = len(power_trace)
    look_around=int(look_around_ms*FS//1000)
    for ev in events:
        center_idx = 0
        
        if center == "peak":
            if event_type == 'FP':
                # find peak power around the event
                center_idx = ev['spike_samp']
                start_l = max(0, center_idx - look_around)
                end_l = min(max_idx, center_idx + look_around)
                snippet_around = power_trace[start_l:end_l] if end_l > start_l else []
                
                if len(snippet_around) > 0:
                    rel_idx = np.argmax(snippet_around)
                    center_idx = start_l + rel_idx
            else:
                # TP or FN: Find peak power in ripple interval
                r_start = int(ev['ripple_start'])
                r_end = int(ev['ripple_end'])
                
                # Clamp
                r_start = max(0, r_start)
                r_end = min(max_idx, r_end)
                
                if r_end > r_start:
                    ripple_power = power_trace[r_start:r_end]
                    if len(ripple_power) > 0:
                        peak_offset = np.argmax(ripple_power)
                        center_idx = r_start + peak_offset
                    else:
                        center_idx = r_start
                else:
                    center_idx = r_start
        elif center=="spike":
            # Center on event (spike or middle of ripple)
            if event_type == 'FN':
                r_start = int(ev['ripple_start'])
                r_end = int(ev['ripple_end'])
                center_idx = int((r_start + r_end) / 2)
            else:
                # TP or FP
                center_idx = ev['spike_samp']
        elif center=="center":
            # Center on event (spike or middle of ripple)
            r_start = int(ev['ripple_start'])
            r_end = int(ev['ripple_end'])
            center_idx = int((r_start + r_end) / 2)

                
        # Extract window
        start = center_idx - window_samp
        end = center_idx + window_samp
        
        if start >= 0 and end <= max_idx:
            snippets.append(power_trace[start:end])
            
    return snippets

def extract_properties(raw_signal,spikified,session,power_trace=None,events=None,event_type=None,center="peak",window_ms=100, fs=30000, look_around_ms=50,network=None, adapt=None,min_resolution=10.0,duration_std=3): 
    look_around=int(look_around_ms*fs//1000)
    max_idx = len(power_trace)
    window_pts = int((window_ms / 1000) * fs)
    half_win = window_pts // 2
    results=[]
    for idx, ev in enumerate(events):
        center_idx = 0      
        if center == "peak":
            if event_type == 'FP':
                # find peak power around the event
                center_idx = ev['spike_samp']
                start_l = max(0, center_idx - look_around)
                end_l = min(max_idx, center_idx + look_around)
                # find duration
                snippet_around = power_trace[start_l:end_l] if end_l > start_l else []
                if len(snippet_around) > 0:
                    rel_idx = np.argmax(snippet_around)
                    max_power = snippet_around[rel_idx]
                    center_idx = start_l + rel_idx
                    
                    threshold = max_power / duration_std
                    above_threshold = power_trace >= threshold
                    # Find continuous regions above threshold
                    diff = np.diff(above_threshold.astype(int))
                    starts = np.where(diff == 1)[0] + 1
                    ends = np.where(diff == -1)[0]
                    if above_threshold[0]:
                        starts = np.concatenate([[0], starts])
                    if above_threshold[-1]:
                        ends = np.concatenate([ends, [len(above_threshold)]])
                    for s, e in zip(starts, ends):
                        if s <= center_idx < e:
                            ev['ripple_start']=s
                            ev['ripple_end']=e
                            break

            else:
                # TP or FN: Find peak power in ripple interval
                r_start = int(ev['ripple_start'])
                r_end = int(ev['ripple_end'])
                
                # Clamp
                r_start = max(0, r_start)
                r_end = min(max_idx, r_end)
                
                if r_end > r_start:
                    ripple_power = power_trace[r_start:r_end]
                    if len(ripple_power) > 0:
                        peak_offset = np.argmax(ripple_power)
                        center_idx = r_start + peak_offset
                    else:
                        center_idx = r_start
                else:
                    center_idx = r_start
        elif center=="spike":
            # Center on event (spike or middle of ripple)
            if event_type == 'FN':
                r_start = int(ev['ripple_start'])
                r_end = int(ev['ripple_end'])
                center_idx = int((r_start + r_end) / 2)
            else:
                # TP or FP
                center_idx = ev['spike_samp']
        elif center=="center":
            # Center on event (spike or middle of ripple)
            r_start = int(ev['ripple_start'])
            r_end = int(ev['ripple_end'])
            center_idx = int((r_start + r_end) / 2)
        # Extract window
        t0 = center_idx - half_win 
        t1 = center_idx + half_win + (window_pts % 2) # Ensure correct length
        if t0 < 0 or t1 >= len(raw_signal):
            continue
        t0_1kHz=int(t0*1000//fs)
        t1_1kHz=int(t1*1000//fs)
        spikified_snippet=spikified[t0_1kHz:t1_1kHz,:]
        spike_number=np.sum(spikified_snippet)

        # Calculate Rate (Hz) -> Count / Duration (s)
        duration_s = (t1 - t0) / fs
        spike_rate = spike_number / duration_s if duration_s > 0 else 0

        spikified_start=int((ev['ripple_start']/fs)*1000)
        spikified_end=int((ev['ripple_end']/fs)*1000)
        spikified_within_event=spikified[spikified_start:spikified_end,:]
        spike_number_within_event=np.sum(spikified_within_event)

        # Calculate Rate Within Event (Hz)
        event_duration_s = (ev['ripple_end'] - ev['ripple_start']) / fs
        spike_rate_within_event = spike_number_within_event / event_duration_s if event_duration_s > 0 else 0
        
        chunk = raw_signal[t0:t1]
                # A) Preferred Frequency

        pref_freq = calculate_preferred_frequency(chunk, fs, band=(70,400),min_resolution=min_resolution)

        # B) <100 Hz Contribution
        low_100_contrib = calculate_low_freq_contribution(chunk, fs, cutoff=100)
        
        # C) Entropy
        entropy = calculate_spectral_entropy(chunk, fs)

        results.append({
            "Session": session,
            "Network": network,
            "Adapt": adapt,
            "Event_Idx": idx,
            "Event_Type": event_type,
            "Duration_ms": event_duration_s * 1000,
            "Preferred_Freq": pref_freq,
            "Low_Freq_Contrib": low_100_contrib,
            "Entropy": entropy,
            "Spike_Number": spike_number,
            "Spike_Rate": spike_rate,
            "Spike_Number_within_Event": spike_number_within_event,
            "Spike_Rate_within_Event": spike_rate_within_event,
            "Event_ID":idx
        })
    return results

def z_score_all(df, metrics=None):
    """
    Z-scores specific columns in the DataFrame.
    Returns a copy to avoid SettingWithCopy warnings.
    """
    df_out = df.copy()
    for col in metrics:
        if col in df_out.columns:
            mean_val = df_out[col].mean()
            std_val = df_out[col].std()
            if std_val != 0:
                df_out[col] = (df_out[col] - mean_val) / std_val
            else:
                df_out[col] = 0
    return df_out

def calculate_covariance_matrix(df, metrics):
    """
    Calculates the covariance matrix for the specified metrics in the DataFrame.
    """
    # Drop NaNs before covariance to avoid errors
    data = df[metrics].dropna().values
    cov_matrix = np.cov(data, rowvar=False)
    return cov_matrix

def pca_from_cov(cov, sort=True):
    """
    Perform PCA given a covariance matrix.
    Returns eigenvalues and eigenvectors.
    """
    # Eigen-decomposition (covariance is symmetric → use eigh)
    eigvals, eigvecs = np.linalg.eigh(cov)

    if sort:
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs

def project_data(df, metrics, eigvecs):
    """
    Project the Z-scored data onto the Principal Components.
    PC_scores = Data_matrix * Eigenvectors
    """
    data = df[metrics].dropna().values
    projected = np.dot(data, eigvecs)
    
    # Return as DataFrame for easy plotting
    # Naming cols PC1, PC2, ...
    cols = [f"PC{i+1}" for i in range(eigvecs.shape[1])]
    projected_df = pd.DataFrame(projected, columns=cols, index=df.dropna(subset=metrics).index)
    
    # Join back with metadata
    result = df.dropna(subset=metrics).copy()
    for col in cols:
        result[col] = projected_df[col]
        
    return result

def plot_evr(evr, title_suffix=""):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(evr)+1), evr/sum(evr), 'o-', linewidth=2)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Scree Plot {title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pca_scatter(df_projected, color_by, title="PCA Projection",evrs=None):
    """
    Scatter plot of PC1 vs PC2, colored by 'color_by'.
    """
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(
        data=df_projected, 
        x="PC1", 
        y="PC2", 
        hue=color_by, 
        alpha=0.7,
        palette="viridis"
    )
    percents=evrs*100/sum(evrs) if evrs is not None else 0
    plt.title(title)
    plt.xlabel(f"PC1 ({percents[0]:.1f}%)" if evrs is not None else "PC1")
    plt.ylabel(f"PC2 ({percents[1]:.1f}%)" if evrs is not None else "PC2")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_loadings(eigvecs, metrics_to_use):
    # Also plot Loading Matrix (Eigenvectors) to understand PCs
    plt.figure(figsize=(10, 5))
    sns.heatmap(eigvecs, annot=True, fmt=".2f", yticklabels=metrics_to_use, xticklabels=[f"PC{i+1}" for i in range(len(metrics_to_use))])
    plt.title(f"PCA Loadings ")
    plt.show()

def run_power_analysis():
    # Configuration
    SPIKES_ROOT = os.path.join(ROOT_DIR, "snnTorch", "generalization_madrid", "spikes")
    DATA_ROOT=os.path.join(ROOT_DIR, "snnTorch", "generalization_madrid", "data")
    os.makedirs(DATA_ROOT, exist_ok=True)
    DATA_PATH_ORIGINAL = r"C:\PedroFelix\Madrid_tests" # Update this if different
    DATA_PATH_EXTRA = r"C:\PedroFelix\extra_data\original_data" # Update this if different
    OUTPUT_FILE = os.path.join(SPIKES_ROOT, "power_analysis_results.pkl")
    center="peak"
    # Analysis parameters
    WINDOW_MS = 100 # +/- 100ms
    FS = 30000
    look_around_ms=50 # for peak centering of FPs
    min_resolution=2.0 # Hz

    SESSIONS_TO_EXCLUDE={        
        "2025-09-24_17-38-17", # Barely any ripples (14 in total...)
        "2025-09-25_12-52-22", # Not a good session to detect ripples
        }
    SESSIONS_TO_EXCLUDE.update(lists_sessions.HFO_sessions) # Exclude HFO Sessions...
    print(f"Searching for spike files in: {SPIKES_ROOT}")
    
    all_power_data = {} # Key: (Network, Adapt), Value: {'TP': [], 'FP': [], 'FN': []}
    results_all=[]
    # Walk through all subdirectories
    for root, dirs, files in os.walk(SPIKES_ROOT):
        for file in files:
            if file.endswith("_spikes.pkl"):
                pkl_path = os.path.join(root, file)
                print(f"\nProcessing: {file}")
                
                # Determine network name and adapt
                net_name = file.replace('_spikes.pkl', '')
                match = re.search(r'_adapt(\d+)$', net_name)
                adapt = int(match.group(1)) if match else 0
                if adapt==20:
                    net_name_clean = re.sub(r'_adapt\d+$', '', net_name) if match else net_name
                    
                    key = (net_name_clean, adapt)
                    if key not in all_power_data:
                        all_power_data[key] = {}
                    
                    # Load spikes
                    with open(pkl_path, 'rb') as f:
                        all_spikes = pickle.load(f)
                    
                    # Iterate over sessions
                    for session, data in tqdm(all_spikes.items(), desc="Sessions"):
                        all_power_data[key][session] = {'TP': [], 'FP': [], 'FN': []}
                        if session in SESSIONS_TO_EXCLUDE:
                            print(f"  Skipping excluded session: {session}")
                            continue
                        else:
                            print(f"🔄Processing session: {session}")

                        channel = data['channel']
                        spikes_ms = np.array(data['spikes'])
                        

                        
                        if session in lists_sessions.extra_sessions:
                            
                            # Load Signal
                            try:
                                liset=liset_tk_extra(data_path=DATA_PATH_EXTRA, name=session, downsample=False, scale_data=False,normalize=False, verbose=False,channels=[channel-1])
                                ripples=liset.ripples_GT
                              
                            except Exception as e:
                                print(f"  Error loading data for {session}: {e}")
                                continue
                        else: 
                            # Load Signal
                            try:
                                liset=read_data(DATA_PATH_ORIGINAL, session, downsample=False, normalize=False, numSamples=False,
                                               start=0, verbose=False,channels=[channel-1])
                                ripples=liset.annotated.ripples_GT
                            except Exception as e:
                                print(f"  Error loading data for {session}: {e}")
                                continue

                        if session in lists_sessions.start_on_40:
                            spikes_ms=spikes_ms[spikes_ms>=40000] # Start on 40s for some sessions with no annotations at the beginning
                            ripples=ripples[ripples[:,0]>=40*liset.fs] # Start on 40s for some sessions with no annotations at the beginning
                        channel_data = liset.data[:,0]
                        filtered_signal=bandpass_filter(channel_data, bandpass=[100,250], fs=liset.fs, order=4) 
                        if filtered_signal is None or ripples is None:
                            continue

                        spikified, _ =spikify_signal(
                                                filtered_signal,
                                                fs=liset.fs,
                                                time_max=20.0,
                                                overlap=0.5,
                                                adapt_threshold=True,
                                                percentile=False,
                                                window_size=0.10,
                                                sample_ratio=0.25,
                                                scaling_factor=1.0,
                                                refractory=0,
                                                factor=30,
                                                initial_value=None,
                                                verbose=False,
                                                ripples=None,     # kept for compatibility but unused
                                            )
                        # Compute Power Trace
                        # filtered_signal is already 100-250Hz
                        power_trace = ripple_band_power_trace(channel_data, liset.fs, smooth_ms=5, zscore=True,bandpass=(100,250))
                        
                        # Extract Events
                        # ripples from load_experimental_data are in samples (30kHz) 
                        tp_ev, fp_ev, fn_ev = extract_events(spikes_ms, ripples, fs=liset.fs,fp_grouping_ms=100, extra_tolerance_ms=20)

                        WINDOW_SAMP=int(WINDOW_MS * liset.fs / 1000)
                        # Extract Snippets
                        tp_snips = extract_power_snippets(power_trace, tp_ev, 'TP', WINDOW_SAMP,center=center,look_around_ms=look_around_ms, FS=liset.fs)
                        fp_snips = extract_power_snippets(power_trace, fp_ev, 'FP', WINDOW_SAMP,center=center,look_around_ms=look_around_ms, FS=liset.fs)
                        fn_snips = extract_power_snippets(power_trace, fn_ev, 'FN', WINDOW_SAMP,center=center,look_around_ms=look_around_ms, FS=liset.fs)
                        results_tp=extract_properties(channel_data, spikified, session, power_trace=power_trace, events=tp_ev, event_type='TP',center=center,window_ms=WINDOW_MS, fs=liset.fs, look_around_ms=look_around_ms,network=net_name_clean, adapt=adapt, min_resolution=min_resolution)
                        results_fp=extract_properties(channel_data, spikified, session, power_trace=power_trace, events=fp_ev, event_type='FP',center=center,window_ms=WINDOW_MS, fs=liset.fs, look_around_ms=look_around_ms,network=net_name_clean, adapt=adapt, min_resolution=min_resolution)
                        results_fn=extract_properties(channel_data, spikified, session, power_trace=power_trace, events=fn_ev, event_type='FN',center=center,window_ms=WINDOW_MS, fs=liset.fs, look_around_ms=look_around_ms,network=net_name_clean, adapt=adapt, min_resolution=min_resolution)
                        results_all.extend(results_tp)
                        results_all.extend(results_fp)
                        results_all.extend(results_fn)
                        
                        # Store (downsample to save space? maybe 1000Hz?)
                        # Let's downsample by factor of 3 (30kHz -> 10kHz)
                        factor = 3
                        all_power_data[key][session]['TP'].extend([s[::factor] for s in tp_snips])
                        all_power_data[key][session]['FP'].extend([s[::factor] for s in fp_snips])
                        all_power_data[key][session]['FN'].extend([s[::factor] for s in fn_snips])
                    
                        print(f"✅Finished session: {session}. TP: {len(tp_snips)}, FP: {len(fp_snips)}, FN: {len(fn_snips)}")

    # Save results
    filename=f"power_analysis_results_center_{center}_{WINDOW_MS}ms.pkl"
    filename_properties=f"power_analysis_properties_center_{center}_{WINDOW_MS}ms.csv"
    df_results=pd.DataFrame(results_all)
    
    OUTPUT_FILE = os.path.join(DATA_ROOT, filename)
    OUTPUT_FILE_PROPERTIES = os.path.join(DATA_ROOT, filename_properties)
    print(f"\nSaving properties results to {OUTPUT_FILE_PROPERTIES}...")
    df_results.to_csv(OUTPUT_FILE_PROPERTIES, index=False)

    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_power_data, f)
    print("Done.")

if __name__ == "__main__":
    run_power_analysis()
