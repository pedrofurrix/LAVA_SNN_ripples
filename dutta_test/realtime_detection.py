"""Realtime ripple-band detection pipeline for single-channel raw LFP.

Functions
- process_and_detect(raw, fs=30000, ...): runs filtering, envelope smoothing
  and threshold-crossing detection returning detection times and stats.

Usage
------
Import the module and call `process_and_detect(raw, fs=30000, alpha=3.0)`
where `raw` is a 1-D numpy array sampled at `fs` Hz.
"""
from typing import Dict, List, Tuple
import numpy as np
from scipy import signal
import os
import pickle as pkl

import sys
curr_dir=os.path.dirname(os.path.abspath(__file__))
par_dir=os.path.dirname(curr_dir)
sys.path.insert(0, par_dir)
from liset_tk.read_data import read_data
from liset_tk.liset_tk_extra import liset_tk_extra
import liset_tk.lists_sessions as lists_sessions
import dutta_test.ripple_filtering as ripple_filtering
from  dutta_test.process_signal import *

def process_and_detect(
    raw: np.ndarray,
    fs: float = 30000.0,
    decim_factor: int = 10,
    bessel_order: int = 2,
    bessel_cutoff: float = 400.0,
    ripple_band: Tuple[float, float] = (150.0, 250.0),
    ripple_taps: int = 30,
    envelope_lpf_taps: int = 33,
    envelope_lpf_cutoff: float = 50.0,
    alpha: float = 3.0,
    lockout_ms: float = 200.0,
    max_detections_per_sec: float = 3.0,
) -> Dict:
    """Process a single-channel raw signal and detect ripple events.

    Steps performed:
    - 2nd-order Bessel low-pass at `bessel_cutoff` Hz (IIR)
    - Downsample by `decim_factor` (anti-alias via polyphase)
    - FIR bandpass (Hamming) for ripple band with `ripple_taps`
    - Absolute value (real-time envelope proxy)
    - FIR low-pass at `envelope_lpf_cutoff` Hz with `envelope_lpf_taps`
    - Compute mean/std of smoothed envelope and detect crossings of
      `mean + alpha*std` with lockout and max-detections-per-second

    Returns dict with: envelope, smoothed_envelope, mu, sigma, threshold,
    detections (list of dicts with `sample` and `time_s`), and `fs_down`.
    """

    print("Processing signal for ripple detection...")
    if raw.ndim != 1:
        raise ValueError("raw must be a 1-D array for a single channel")

    # 1) Bessel low-pass IIR
    b_b, a_b = signal.bessel(bessel_order, bessel_cutoff, btype="low", fs=fs)
    sig_lp = signal.filtfilt(b_b, a_b, raw)

    # 2) Downsample (use polyphase resampling for anti-aliasing)
    sig_ds = signal.resample_poly(sig_lp, up=1, down=decim_factor)
    fs_down = fs / decim_factor

    # 3) Ripple band FIR bandpass (Hamming window)
    taps_bp = signal.firwin(ripple_taps, [ripple_band[0], ripple_band[1]], pass_zero=False, fs=fs_down, window="hamming")
    ripple_sig = signal.filtfilt(taps_bp, [1.0], sig_ds)

    # 4) Envelope (real-time proxy: absolute value)
    envelope = np.abs(ripple_sig)

    # 5) Smooth envelope with low-pass FIR
    taps_env = signal.firwin(envelope_lpf_taps, envelope_lpf_cutoff, fs=fs_down, window="hamming")
    smoothed = signal.filtfilt(taps_env, [1.0], envelope)

    # 6) Stats and threshold
    mu = float(np.mean(smoothed))
    sigma = float(np.std(smoothed))
    threshold = mu + alpha * sigma

    # 7) Find upward crossings
    crossings = np.where((smoothed[:-1] < threshold) & (smoothed[1:] >= threshold))[0] + 1

    # 8) Enforce lockout and max detections/sec
    lockout_samps = int(round(lockout_ms / 1000.0 * fs_down))
    detections: List[Dict] = []
    last_detection_idx = -np.inf
    for idx in crossings:
        if idx <= last_detection_idx + lockout_samps:
            continue
        # time of candidate in seconds
        t = idx / fs_down
        # count detections in last 1 second
        recent_count = sum(1 for d in detections if d["time_s"] >= t - 1.0)
        if recent_count >= max_detections_per_sec:
            continue
        detections.append({"sample": int(idx), "time_s": float(t)})
        last_detection_idx = idx

    return {
        "fs_down": fs_down,
        "envelope": envelope,
        "smoothed_envelope": smoothed,
        "mu": mu,
        "sigma": sigma,
        "threshold": float(threshold),
        "detections": detections,
    }


def save_detections_dict(detections_dict: Dict,alpha: float) -> str:
    """Save or append a dictionary of per-session detections to a pickle file.

    detections_dict: mapping session -> detection info (serializable)
    prefix: filename prefix (used to name the pkl)
    out_root: optional directory to place `spikes/<prefix>/`; if None uses module dir

    Returns path to written file.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(curr_dir, "spikes",)
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"detections_thr_{alpha}.pkl")

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                existing = pkl.load(f)
            if isinstance(existing, dict):
                existing.update(detections_dict)
                detections_dict = existing
        except Exception:
            # if any error, we'll overwrite with new dict
            pass

    with open(pkl_path, "wb") as f:
        pkl.dump(detections_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(f"Saved detections to {pkl_path}")
    return pkl_path


def detect_and_save_sessions(
    path:str,
    sessions:set,
    fs: float = 30000.0,
    alpha: float = 3.0,
    **process_kwargs,
) -> str:
    """Run detection for multiple sessions and save results per session.

    sessions_raw: mapping session_id -> raw 1-D numpy array sampled at `fs`.
    prefix: used for output folder and pickle filename.
    process_kwargs: additional kwargs passed to `process_and_detect`.

    Returns path to saved pickle file.
    """
    results = {}
    for session in sessions:
        channel_sessions=lists_sessions.channel_sessions
        extra_sessions=lists_sessions.extra_sessions
        channel=channel_sessions.get(session,None)
        print(f"Processing session {session}, channel {channel}...")
        
        if session in extra_sessions:
            data,ripples=load_experimental_data(path,session, downsample = False, normalize = True, numSamples = False, 
                           start = 0, verbose=True, channel=channel,load_data=True,data_reader=liset_tk_extra)

        else:
            data,ripples=load_experimental_data(path,session, downsample = False, normalize = True, numSamples = False, 
                           start = 0, verbose=True, channel=channel,load_data=True,data_reader=read_data)        
        out = process_and_detect(data, fs=fs, alpha=alpha, **process_kwargs)
        # store compact serializable summary
        results[session] = {
            "fs_down": out["fs_down"],
            "mu": out["mu"],
            "sigma": out["sigma"],
            "threshold": out["threshold"],
            "detections": out["detections"],
        }

    pkl_path = save_detections_dict(results,alpha)
    return pkl_path


if __name__ == "__main__":
    path=r"C:\PedroFelix\extra_data\original_data"
    # session_set={"2025-09-22_17-55-26", #R
    #             "2025-09-23_15-50-26", #R
    #             "2025-09-24_10-24-40", #R
    #             "2025-09-24_14-22-55", #H
    #             "2025-09-24_15-13-10", #H
    #             "2025-09-25_16-41-14"} #R

    # # Extra
    # #     session_set={"2025-09-24_16-29-07", #R   
    # #             "2025-09-24_17-38-17",} #R 
    # session_set.update({ 
    #     "2025-09-24_16-29-07", #R   
    #     "2025-09-24_17-38-17", #R 
    #     "2025-09-22_17-42-27", 
    #     "2025-09-23_16-17-52", 
    #     "2025-09-24_11-34-51",
    #     "2025-09-25_11-21-53",
    #     "2025-09-25_12-52-22",})
    session_set=lists_sessions.extra_sessions
    alphas=[3.0,4.0,5.0,6.0,7.0,8.0]
    for alpha in alphas:
        detect_and_save_sessions(
            path,
            session_set,
            fs=30000,
            alpha=alpha,
        )