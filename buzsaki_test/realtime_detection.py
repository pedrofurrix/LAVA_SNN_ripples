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
from liset_data_reader.read_data import read_data
from liset_data_reader.liset_tk_extra import liset_tk_extra
import liset_data_reader.lists_sessions as lists_sessions

from  buzsaki_test.process_signal import load_experimental_data
from buzsaki_test.gt_annotations import *


def save_detections_dict(detections_dict: Dict,alpha1: float,alpha2: float) -> str:
    """Save or append a dictionary of per-session detections to a pickle file.

    detections_dict: mapping session -> detection info (serializable)
    prefix: filename prefix (used to name the pkl)
    out_root: optional directory to place `spikes/<prefix>/`; if None uses module dir

    Returns path to written file.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(curr_dir, "spikes",)
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"detections_thr_{alpha1}_{alpha2}.pkl")

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
    path_og:str,
    path_extra:str,
    sessions:set,
    fs: float = 1250,
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
            print("Loading from extra data reader...")
            data,ripples=load_experimental_data(path_extra,session, downsample = fs, normalize = False, numSamples = False, 
                           start = 0, verbose=True, channel=channel,load_data=True,data_reader=liset_tk_extra,scale_data=True)

        else:
            print("Loading from Madrid data reader...")
            data,ripples=load_experimental_data(path_og,session, downsample = fs, normalize = False, numSamples = False, 
                           start = 0, verbose=True, channel=channel,load_data=True,data_reader=read_data,scale_data=True)        
        out_pd=buzsaki(data,fs=fs,**process_kwargs)
        # store compact serializable summary
        results[session] = {
            "start_s": out_pd["start_s"].tolist(),
            "end_s": out_pd["end_s"].tolist(), 
            "detection_times": out_pd["thresh_cross_time"].tolist(),
        }

    pkl_path = save_detections_dict(results,process_kwargs["threshold"],process_kwargs["fall_off"])
    return pkl_path


if __name__ == "__main__":
    path_extra=r"C:\PedroFelix\extra_data\original_data"
    path_og=r"C:\PedroFelix\Madrid_tests"
    # session_set=lists_sessions.extra_sessions
    session_set={"2025-09-22_17-55-26", #R
                    "2025-09-23_15-50-26", #R
                    "2025-09-24_10-24-40", #R
                    "2025-09-25_16-41-14",
                    "2025-09-24_16-29-07", #R   
                    "2025-09-24_17-38-17",
                    "2025-09-24_16-29-07", #R   
                    "2025-09-24_17-38-17", #R 
                    "2025-09-22_17-42-27", 
                    "2025-09-23_16-17-52", 
                    "2025-09-24_11-34-51",
                    "2025-09-25_11-21-53",
                    "2025-09-25_12-52-22",
                    } #R

    print(session_set)
    # alphas=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    alphas=[2.5]
    fall_offs=[2.0]
    # fall_offs=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
    for alpha in alphas:
        for fall_off in fall_offs:
            if fall_off < alpha:
                detect_and_save_sessions(
                    path_og,
                    path_extra,
                    session_set,
                    fs=1250,
                    threshold=alpha,
                    fall_off=fall_off
                )