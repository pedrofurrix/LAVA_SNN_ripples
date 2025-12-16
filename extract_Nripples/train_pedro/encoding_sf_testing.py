import sys
import os
# liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liset_tk'))
curr_path=os.path.abspath(os.path.dirname(__file__))
project_dir=os.path.join(curr_path, os.pardir, os.pardir,os.pardir)

sys.path.insert(0, project_dir)

# from liset_tk.liset_aux import ripples_std, middle
from liset_tk.signal_aid import bandpass_filter,highpass_filter

from extract_Nripples.utils_encoding import *
import matplotlib.pyplot as plt
# from liset_tk import liset_tk
from liset_tk.liset_paper import liset_paper as liset_tk

import os
import numpy as np
from copy import deepcopy
import time
import json
from matplotlib.lines import Line2D

#### LAB PC
# parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN_TRAINING_SESSIONS" # Modify this to your data path folder
# parent = r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\Download_from_paper" # Modify this to your data path folder

### HOME PC
# parent=r"E:\neurospark_mat\CNN_TRAINING_SESSIONS"
parent=r"E:\neurospark_mat\Download_from_paper"

downsampled_fs= 30000
save_dir = os.path.join(os.path.dirname(__file__))
time_max=10 # seconds
window_size=0.1 # seconds # 50 ms
sample_ratio=0.25 # ratio of max amplitudes to use
scaling_factor=0.5 # scale factor for the threshold
refractory=0.0003 # seconds
bandpass=[100,250]
min_threshold=0.1 # minimum threshold for the spike detection
save=True

factor=30

def make_up_down_data(save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs, sample_ratio=sample_ratio,scaling_factor=scaling_factor,
                        time_max=time_max,window_size=window_size, refractory=refractory,
                      parent=parent,save=save,verbose=False,factor=factor,adapt_threshold=False,overlap=0.5):
    metrics={}
    metrics["parameters"]={
            "bandpass": bandpass,
            "downsampled_fs": downsampled_fs,
            "sample_ratio": sample_ratio,
            "scaling_factor": scaling_factor,
            "time_max": time_max,
            "window_size": window_size,
            "refractory": refractory,
            "factor": factor,
            "adapt_threshold": adapt_threshold,
            "overlap": overlap
            }
    
    ############## Load the data ##################
    datasets=os.listdir(parent)
    if save:
        save_path=os.path.join(save_dir,f"{int(scaling_factor*10)}") if not adapt_threshold else os.path.join(save_dir,f"adapt_{time_max}",f"{int(scaling_factor*10)}")

    for dataset in datasets:

        metrics[dataset]={}
        thresholds=[]
        lost_spikes=[]
        total_spikes=[]

        dataset_path=os.path.join(parent,dataset)
        print("Dataset:",dataset)
        save_dataset_path=os.path.join(save_path,dataset)
        liset=liset_tk(dataset_path, shank=1, downsample=False, start=0, verbose=False)
            
        factor_downsample=liset.fs//downsampled_fs
        overall_factor=factor*factor_downsample   
        ripples=liset.ripples_GT//overall_factor
        if verbose:
            print(f"Frequency: {downsampled_fs} Hz")
            print("Loaded LFPs:",dataset_path)

        filtered_liset=np.zeros((int(liset.data.shape[0]//factor_downsample),int(liset.data.shape[1])))
        up_down=np.zeros((int(liset.data.shape[0]//overall_factor),int(liset.data.shape[1]),2)) # 2 for UP and DOWN spikes

        channels=[i for i in range(liset.data.shape[1])]
        for channel in channels:
            if verbose:
                print("Channel:", channel+1)
            channel_filtered=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
            if factor_downsample>1:
                channel_filtered=decimation_downsampling(channel_filtered,factor_downsample)
                
            filtered_liset[:,channel]=channel_filtered
            if verbose:
                print("Downsampling factor:",factor_downsample)

            spikified,thr,spikes_lost=spikify_signal(
                channel_filtered,
                fs=downsampled_fs,
                time_max=time_max,
                overlap=overlap,
                adapt_threshold=adapt_threshold,
                percentile=False,
                window_size=window_size,
                sample_ratio=sample_ratio,
                scaling_factor=scaling_factor,
                refractory=refractory,
                factor=factor,
                initial_value=None,
                verbose=verbose,
                ripples=ripples
            )

            if factor>1:
                lost_spikes.append(spikes_lost)
                spikes_total= np.sum(spikified)
                total_spikes.append(spikes_total)
                thresholds.append(thr)
            up_down[:,channel,:]= spikified
  
        metrics[dataset]["channels"]=channels
        metrics[dataset]["thresholds"]=thresholds
        metrics[dataset]["spikes_lost"]=lost_spikes
        metrics["parameters"]["factor_downsample"]=factor_downsample    
        if save:
            os.makedirs(save_dataset_path, exist_ok=True)
            np.save(os.path.join(save_dataset_path, f'up_down.npy'), up_down)
            print(f"UP/DOWN data saved in {os.path.join(save_dataset_path, 'up_down.npy')}")

            
    if save:
        with open(os.path.join(save_path, f'parameters.json'), 'w') as f:
            json.dump(metrics,f, indent=4)  # Save parameters
        print(f"Parameters saved in {os.path.join(save_path, 'parameters.json')}")

        print(f'Shape of the filtered data: {filtered_liset.shape}')
        print(f'Shape of the UP/DN data: {up_down.shape}')

def spikify_signal(
    signal,
    fs,
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
):

    N = len(signal)
    out_len = N // factor if factor > 1 else N
    spikified = np.zeros((out_len, 2))

    win = int(fs * time_max)
    step = int(fs * overlap * time_max)

    # -------- THRESHOLD HELPERS --------
    def compute_threshold(x):
        return calculate_threshold(x, fs, window_size, sample_ratio, scaling_factor)

    def get_threshold_window(signal, t):
        """Sliding window for adaptive threshold."""
        if t < win:
            return signal[:win]
        else:
            return signal[t - win:t]

    if verbose:
        print(f"[spikify] N={N}, out_len={out_len}, win={win}, step={step}, factor={factor}")

    # =============================
    # ADAPTIVE THRESHOLD MODE
    # =============================
    if adapt_threshold:

        thresholds = []

        for t in range(0, N, step):

            # --- sliding threshold window ---
            tw = get_threshold_window(signal, t)

            thr = compute_threshold(tw)
            thresholds.append(thr)

            # --- extract chunk ---
            r_edge = min(t + step, N)
            chunk = signal[t:r_edge]

            # --- spiking ---
            spk, initial_value = up_down_channel(
                chunk, thr, fs, refractory,
                initial_value=initial_value, return_value=True
            )

            # Downsample spikes if using factor
            if factor > 1:
                spk, lost_spikes = extract_spikes_downsample(spk, factor)

            L = t // factor
            R = r_edge // factor
            spikified[L:R] = spk

        if verbose:
            print("Spikification complete.")
            print(f"UP spikes:   {np.sum(spikified[:,0])}")
            print(f"DOWN spikes: {np.sum(spikified[:,1])}")

        return spikified, thresholds, lost_spikes

    # =============================
    # FIXED THRESHOLD MODE
    # =============================
    else:
        thr = compute_threshold(signal[:win])

        spk = up_down_channel(signal, thr, fs, refractory,
                              initial_value=None, return_value=False)

        if factor > 1:
            spk, lost_spikes = extract_spikes_downsample(spk, factor)

        return spk, thr,lost_spikes

def evaluate_encoding(save_dir=save_dir,scaling_factor=scaling_factor,
                      parent=parent,save=save,highpass=False,verbose=False,adapt_threshold=False,time_max=time_max):
    """
    
    Evaluate the encoding of the UP/DOWN spikes

    """
    metrics={}
    total_up_all = 0
    total_down_all = 0
    global_metrics_sum = {
        "SNR": [],
        "RMSE": [],
        "R_squared": [],
        "AFR": [],
        "SNR_ripples": [],
        "RMSE_ripples": [],
        "R_squared_ripples": [],
        "AFR_ripples": [],
        "SNR_non_ripples": [],
        "RMSE_non_ripples": [],
        "R_squared_non_ripples": [],
        "AFR_non_ripples": []
    }

    path=os.path.join(save_dir,f"{int(scaling_factor*10)}") if not adapt_threshold else os.path.join(save_dir,f"adapt_{time_max}",f"{int(scaling_factor*10)}")

    with open(os.path.join(path, 'parameters.json'), 'r') as f:
        parameters = json.load(f)

    factor=parameters["parameters"]["factor"]
    factor_downsample=parameters["parameters"]["factor_downsample"]
    downsampled_fs=parameters["parameters"]["downsampled_fs"]
    for dataset in parameters:
        if dataset == "parameters":
            continue
        metrics[dataset]={}
        dataset_path=os.path.join(parent,dataset)
        overall_factor=factor*factor_downsample

        liset=liset_tk(dataset_path, shank=1, downsample=False, start=0, verbose=False)
        ripples=liset.ripples_GT//overall_factor
        filtered_liset=np.zeros((int(liset.data.shape[0]//overall_factor),int(liset.data.shape[1])))
        
        channels=[i for i in range(liset.data.shape[1])]
        for channel in channels:
        
            if verbose:
                print("Channel:", channel+1)

            channel_filtered=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
           
            if overall_factor>1:
                filtered_liset[:,channel]=decimation_downsampling(channel_filtered,overall_factor)
                # filtered_liset[:,channel]=average_downsampling(channel_filtered,factor)
            else:
                filtered_liset[:,channel]=channel_filtered
            if verbose:
                print("Downsampling factor:",overall_factor)   

        up_down_path=os.path.join(path,dataset, f"up_down.npy")
        up_down=np.load(up_down_path)
   
    
        thresholds=parameters[dataset]["thresholds"]
        # Reconstruct the signal
        reconstructed_signal=np.zeros((up_down.shape[0],up_down.shape[1]))
        
        

        # Get parameters for adaptive threshold
        overlap = parameters["parameters"].get("overlap", 0.5)
        param_time_max = parameters["parameters"].get("time_max", time_max)

        for channel in channels:
            metrics[dataset][channel]={}
            reconstructed_signal[0,channel]=filtered_liset[0,channel]
            
            if adapt_threshold:
                step = int(downsampled_fs * overlap * param_time_max)

            # Loop through the up_down data and reconstruct the signal
            for t in range(1, up_down.shape[0]):
                if adapt_threshold:
                    thr_idx = int((t * factor) / step)
                    if thr_idx >= len(thresholds[channel]):
                        thr_idx = len(thresholds[channel]) - 1
                    current_threshold = thresholds[channel][thr_idx]
                else:
                     current_threshold = thresholds[channel]

                spike_plus = up_down[t, channel,0]
                spike_minus = up_down[t, channel,1]
                if spike_plus == 1:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel] + current_threshold
                elif spike_minus == 1:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel] - current_threshold
                else:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel]
            if highpass:
                reconstructed_signal[:, channel] = highpass_filter(reconstructed_signal[:, channel], fs=downsampled_fs//factor, highpass=highpass)

        # Calculate error metrics between the original and reconstructed signal
        for channel in channels:
            print("Channel:", channel+1)
            s_full = filtered_liset[:, channel]
            r_full = reconstructed_signal[:, channel]
            spikes_full = up_down[:, channel, 0] + up_down[:, channel, 1]

            # Calculate metrics
            metrics[dataset][channel]["general"]={
                "SNR": calculate_snr(s_full, r_full),
                "RMSE": calculate_rmse(s_full, r_full),
                "R_squared": calculate_r_squared(s_full, r_full),
                "AFR": calculate_average_spike_rate(spikes_full,downsampled_fs//factor)
            }

            if verbose:
                print("General Metrics:\n",metrics[dataset][channel]["general"]) 
            # Calculate metrics for ripples
            # --- Ripple metrics (average across ripple windows)
            snrs, rmses, r2s, afrs = [], [], [], []

            for ripple in ripples:
                start, end = ripple[0], ripple[1]
                s = s_full[start:end]
                r = r_full[start:end]
                spikes = spikes_full[start:end]

                snrs.append(calculate_snr(s, r))
                rmses.append(calculate_rmse(s, r))
                r2s.append(calculate_r_squared(s, r))
                afrs.append(calculate_average_spike_rate(spikes,downsampled_fs//factor))

            # Store averaged ripple metrics
            if snrs:  # in case ripples is empty
                metrics[dataset][channel]["ripples"] = {
                    "SNR": float(np.mean(snrs)),
                    "RMSE": float(np.mean(rmses)),
                    "R_squared": float(np.mean(r2s)),
                    "AFR": float(np.mean(afrs))
                }
            else:
                metrics[dataset][channel]["ripples"] = {
                    "SNR": None,
                    "RMSE": None,
                    "R_squared": None,
                    "AFR": None
                }
            print("Ripple Metrics:\n",metrics[dataset][channel]["ripples"])

            # --- Non-ripple metrics ---
            non_ripple_mask = np.ones(len(s_full), dtype=bool)
            for ripple in ripples:
                start, end = ripple[0], ripple[1]
                non_ripple_mask[start:end] = False  # mask out ripple periods

            # Masked data
            s_non_ripple = s_full[non_ripple_mask]
            r_non_ripple = r_full[non_ripple_mask]
            spikes_non_ripple = spikes_full[non_ripple_mask]

            metrics[dataset][channel]["non_ripples"] = {
                "SNR": calculate_snr(s_non_ripple, r_non_ripple),
                "RMSE": calculate_rmse(s_non_ripple, r_non_ripple),
                "R_squared": calculate_r_squared(s_non_ripple, r_non_ripple),
                "AFR": calculate_average_spike_rate(spikes_non_ripple, downsampled_fs // factor)
            }

        total_up_spikes = int(np.sum(up_down[:, :, 0]))
        total_down_spikes = int(np.sum(up_down[:, :, 1]))
        if verbose:
            print("Total Dataset:",dataset)
            print("Total UP Spikes:", total_up_spikes)
            print("Total DOWN Spikes:", total_down_spikes)
            print("Total Spikes:", total_up_spikes + total_down_spikes)

        metrics[dataset]["total_up_spikes"] = total_up_spikes
        metrics[dataset]["total_down_spikes"] = total_down_spikes
        metrics[dataset]["total_spikes"] = total_up_spikes + total_down_spikes
        
        metrics[dataset]["average_channels"]={
            "SNR": float(np.mean([metrics[dataset][channel]["general"]["SNR"] for channel in channels])),
            "RMSE": float(np.mean([metrics[dataset][channel]["general"]["RMSE"] for channel in channels])),
            "R_squared": float(np.mean([metrics[dataset][channel]["general"]["R_squared"] for channel in channels])),
            "AFR": float(np.mean([metrics[dataset][channel]["general"]["AFR"] for channel in channels]))
        }
        metrics[dataset]["average_ripples"]={
            "SNR": float(np.mean([metrics[dataset][channel]["ripples"]["SNR"] for channel in channels])),
            "RMSE": float(np.mean([metrics[dataset][channel]["ripples"]["RMSE"] for channel in channels])),
            "R_squared": float(np.mean([metrics[dataset][channel]["ripples"]["R_squared"] for channel in channels])),
            "AFR": float(np.mean([metrics[dataset][channel]["ripples"]["AFR"] for channel in channels]))
        }
        metrics[dataset]["average_non_ripples"] = {
        "SNR": float(np.mean([metrics[dataset][ch]["non_ripples"]["SNR"] for ch in channels])),
        "RMSE": float(np.mean([metrics[dataset][ch]["non_ripples"]["RMSE"] for ch in channels])),
        "R_squared": float(np.mean([metrics[dataset][ch]["non_ripples"]["R_squared"] for ch in channels])),
        "AFR": float(np.mean([metrics[dataset][ch]["non_ripples"]["AFR"] for ch in channels]))
        }
        if verbose:
            print("Average Channels Metrics:\n",metrics[dataset]["average_channels"])
            print("Average Ripples Metrics:\n",metrics[dataset]["average_ripples"])
        # Accumulate UP/DOWN spike counts
        total_up_all += total_up_spikes
        total_down_all += total_down_spikes
        
        # Accumulate channel-level averages
        global_metrics_sum["SNR"].append(metrics[dataset]["average_channels"]["SNR"])
        global_metrics_sum["RMSE"].append(metrics[dataset]["average_channels"]["RMSE"])
        global_metrics_sum["R_squared"].append(metrics[dataset]["average_channels"]["R_squared"])
        global_metrics_sum["AFR"].append(metrics[dataset]["average_channels"]["AFR"])

        # Accumulate ripple-level averages
        global_metrics_sum["SNR_ripples"].append(metrics[dataset]["average_ripples"]["SNR"])
        global_metrics_sum["RMSE_ripples"].append(metrics[dataset]["average_ripples"]["RMSE"])
        global_metrics_sum["R_squared_ripples"].append(metrics[dataset]["average_ripples"]["R_squared"])
        global_metrics_sum["AFR_ripples"].append(metrics[dataset]["average_ripples"]["AFR"])

        global_metrics_sum["SNR_non_ripples"].append(metrics[dataset]["average_non_ripples"]["SNR"])
        global_metrics_sum["RMSE_non_ripples"].append(metrics[dataset]["average_non_ripples"]["RMSE"])
        global_metrics_sum["R_squared_non_ripples"].append(metrics[dataset]["average_non_ripples"]["R_squared"])
        global_metrics_sum["AFR_non_ripples"].append(metrics[dataset]["average_non_ripples"]["AFR"])

    overall_total_spikes = total_up_all + total_down_all
    overall_metrics = {
        "total_spikes": overall_total_spikes,
        "total_up_spikes": total_up_all,
        "total_down_spikes": total_down_all,
        "average_channels": {
            "SNR": float(np.mean(global_metrics_sum["SNR"])),
            "RMSE": float(np.mean(global_metrics_sum["RMSE"])),
            "R_squared": float(np.mean(global_metrics_sum["R_squared"])),
            "AFR": float(np.mean(global_metrics_sum["AFR"]))
        },
        "average_ripples": {
            "SNR": float(np.mean(global_metrics_sum["SNR_ripples"])),
            "RMSE": float(np.mean(global_metrics_sum["RMSE_ripples"])),
            "R_squared": float(np.mean(global_metrics_sum["R_squared_ripples"])),
            "AFR": float(np.mean(global_metrics_sum["AFR_ripples"]))
        },
        "average_non_ripples": {
            "SNR": float(np.mean(global_metrics_sum["SNR_non_ripples"])),
            "RMSE": float(np.mean(global_metrics_sum["RMSE_non_ripples"])),
            "R_squared": float(np.mean(global_metrics_sum["R_squared_non_ripples"])),
            "AFR": float(np.mean(global_metrics_sum["AFR_non_ripples"]))
        }
    }
    if verbose:
        print("Overall Metrics:\n", overall_metrics)
    metrics["overall_metrics"] = overall_metrics

    # Save the metrics
    if save:
        ############## Save the metrics ##################	
        metrics_path=os.path.join(path,f"metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)  # optional: indent=4 for readability
        print(f"Metrics saved in {metrics_path}")
    else:
        print("Metrics not saved, set save=True to save the metrics")
        print("Metrics:", metrics)

def save_reconstruction(scaling_factor=scaling_factor, save_dir=save_dir, parent=parent, highpass=False, verbose=False,adapt_threshold=False,time_max=time_max):
    path=os.path.join(save_dir,f"{int(scaling_factor*10)}") if not adapt_threshold else os.path.join(save_dir,f"adapt_{time_max}",f"{int(scaling_factor*10)}")

    with open(os.path.join(path, 'parameters.json'), 'r') as f:
        parameters = json.load(f)

    factor=parameters["parameters"]["factor"]
    factor_downsample=parameters["parameters"]["factor_downsample"]
    downsampled_fs=parameters["parameters"]["downsampled_fs"]
    for dataset in parameters:
        if dataset == "parameters":
            continue
        dataset_path=os.path.join(parent,dataset)
        overall_factor=factor*factor_downsample

        liset=liset_tk(dataset_path, shank=1, downsample=False, start=0, verbose=False)
        ripples=liset.ripples_GT//overall_factor
        filtered_liset=np.zeros((int(liset.data.shape[0]//overall_factor),int(liset.data.shape[1])))
        
        channels=[i for i in range(liset.data.shape[1])]
        for channel in channels:
        
            if verbose:
                print("Channel:", channel+1)

            channel_filtered=bandpass_filter(liset.data[:,channel], bandpass=bandpass, fs=liset.fs)
           
            if overall_factor>1:
                filtered_liset[:,channel]=decimation_downsampling(channel_filtered,overall_factor)
                # filtered_liset[:,channel]=average_downsampling(channel_filtered,factor)
            else:
                filtered_liset[:,channel]=channel_filtered
            if verbose:
                print("Downsampling factor:",overall_factor)   

        up_down_path=os.path.join(path,dataset, f"up_down.npy")
        up_down=np.load(up_down_path)
        thresholds=parameters[dataset]["thresholds"]
        # Reconstruct the signal
        reconstructed_signal=np.zeros((up_down.shape[0],up_down.shape[1]))
        # Get parameters for adaptive threshold
        overlap = parameters["parameters"].get("overlap", 0.5)
        param_time_max = parameters["parameters"].get("time_max", time_max)

        for channel in channels:
            reconstructed_signal[0,channel]=filtered_liset[0,channel]
            
            if adapt_threshold:
                 step = int(downsampled_fs * overlap * param_time_max)

            # Loop through the up_down data and reconstruct the signal
            for t in range(1, up_down.shape[0]):
                if adapt_threshold:
                    thr_idx = int((t * factor) / step)
                    if thr_idx >= len(thresholds[channel]):
                        thr_idx = len(thresholds[channel]) - 1
                    current_threshold = thresholds[channel][thr_idx]
                else:
                    current_threshold = thresholds[channel]

                spike_plus = up_down[t, channel,0]
                spike_minus = up_down[t, channel,1]
                if spike_plus == 1:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel] + current_threshold
                elif spike_minus == 1:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel] - current_threshold
                else:
                    reconstructed_signal[t, channel] = reconstructed_signal[t - 1, channel]
            if highpass:
                reconstructed_signal[:, channel] = highpass_filter(reconstructed_signal[:, channel], fs=downsampled_fs//factor, highpass=highpass)

        # Save the reconstructed signal
        filename=f"reconstructed_signal_{highpass}Hz.npy" if highpass else "reconstructed_signal.npy"
        save_reconstructed_path = os.path.join(path, dataset, filename)
        np.save(save_reconstructed_path, reconstructed_signal)
        print(f"Reconstructed signal saved in {save_reconstructed_path}")

if __name__ == "__main__":
    
    parent=r"D:\neurospark_mat\Download_from_paper"
    downsampled_fs= 30000
    save_dir = os.path.join(os.path.dirname(__file__))
    time_max=20 # seconds
    window_size=0.1 # seconds # 50 ms
    sample_ratio=0.25 # ratio of max amplitudes to use
    refractory=0.0003 # seconds
    bandpass=[100,250]
    save=True
    factor=30
    highpass=100
    overlap=0.5
    adapt_threshold=True
    scaling_factor = 1.0  # Default value, can be overridden by command line argument
    # args = parse_args()
    # scaling_factor=args.sf
    print(f"Using scaling factor: {scaling_factor}")
    # make_up_down_data(save_dir=save_dir,bandpass=bandpass,downsampled_fs=downsampled_fs, sample_ratio=sample_ratio,scaling_factor=scaling_factor,
    #                     time_max=time_max,window_size=window_size, refractory=refractory,
    #                   parent=parent,save=save,verbose=False,factor=factor,adapt_threshold=adapt_threshold,overlap=overlap)

    # evaluate_encoding(save_dir=save_dir,scaling_factor=scaling_factor,
    #                   parent=parent,save=save,highpass=highpass,verbose=False,adapt_threshold=adapt_threshold,time_max=time_max)
    save_reconstruction(scaling_factor=scaling_factor, save_dir=save_dir, parent=parent, highpass=False, verbose=False,adapt_threshold=adapt_threshold,time_max=time_max)