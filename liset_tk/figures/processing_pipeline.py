import sys
import os
curr_dir=os.path.abspath(os.path.dirname(__file__))
pardir=os.pardir
sys.path.append(os.path.join(curr_dir, pardir))
sys.path.append(os.path.join(curr_dir, pardir, pardir))
from liset_paper import liset_paper as liset_tk

import numpy as np
import matplotlib.pyplot as plt

from signal_aid import bandpass_filter
import matplotlib.pyplot as plt
# import seaborn as sns

from extract_Nripples.utils_encoding import *


parent=r"E:\neurospark_mat\Download_from_paper"
datasets=os.listdir(parent)
print("Available datasets:")
for i, dataset in enumerate(datasets):
    print(f"{i}: {dataset}")
print()
dataset=int(input("Dataset to process: "))
dataset_path=os.path.join(parent,datasets[dataset])


liset=liset=liset_tk(dataset_path, shank=1, downsample=False, start=0, verbose=False)
ripples=liset.ripples_GT
# ripples=np.argsort(ripples, axis=0)


channel=1
window=[5,6] # in seconds
print(ripples[:10])
mask = (ripples[:, 1] >= window[0]*liset.fs) & (ripples[:, 0] <= window[1]*liset.fs)
window_ripples = ripples[mask]
print(f"Ripples in window {window}: {len(window_ripples)}")
channel_data = liset.data[:, channel]


# Setup shared time window
start_idx = window[0] * liset.fs
end_idx = window[1] * liset.fs
time = np.arange(start_idx, end_idx) / liset.fs

# Process Data
filtered_channel = bandpass_filter(channel_data, fs=liset.fs, bandpass=(100,250), order=4)
threshold=calculate_threshold(filtered_channel[:120*liset.fs],liset.fs,0.1,0.25,1,plot=False,verbose=False)
print(f"Threshold: {threshold}")
up_down=up_down_channel(filtered_channel,threshold,liset.fs,refractory=0,initial_value=None,return_value=False)
spikes_downsample,spikes_lost=extract_spikes_downsample(up_down,30,)

# Extract relevant slices
raw_data = channel_data[start_idx:end_idx]
filtered_data = filtered_channel[start_idx:end_idx]
up_down_window = up_down[start_idx:end_idx, :]
up_mask = up_down_window[:, 0] > 0
down_mask = up_down_window[:, 1] > 0
up_spike_times = time[up_mask]
down_spike_times = time[down_mask]

# Downsample window
window_downsample = [5.35, 5.45]
start_ds = int(window_downsample[0] * liset.fs)
end_ds = int(window_downsample[1] * liset.fs)
time_ds = np.arange(start_ds, end_ds) / liset.fs
time_ds_down = np.arange(int(window_downsample[0]*1000), int(window_downsample[1]*1000)) / 1000

up_down_window_ds = up_down[start_ds:end_ds, :]
spikes_ds_window = spikes_downsample[int(window_downsample[0]*1000):int(window_downsample[1]*1000), :]

up_spikes_ds = time_ds[up_down_window_ds[:, 0] > 0]
down_spikes_ds = time_ds[up_down_window_ds[:, 1] > 0]
up_spikes_down = time_ds_down[spikes_ds_window[:, 0] > 0]
down_spikes_down = time_ds_down[spikes_ds_window[:, 1] > 0]

# Filter ripple window for zoom panels
ripple_mask = (window_ripples[:, 1] / liset.fs >= window_downsample[0]) & \
              (window_ripples[:, 0] / liset.fs <= window_downsample[1])
zoom_ripples = window_ripples[ripple_mask]

# Create layout
fig = plt.figure(figsize=(10, 10), dpi=300)
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

# --- Row 1: Raw LFP ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time, raw_data, color='black', linewidth=1)
for i, ripple in enumerate(window_ripples):
    start, end = ripple / liset.fs
    ax1.axvspan(start, end, color='yellow', alpha=0.3, label="GT Ripple" if i == 0 else None)
ax1.set_title("A. Wideband LFP (1 Hz – 5 kHz)")
ax1.set_ylabel("Amplitude")
ax1.legend(loc="best")

# --- Row 2: Filtered ---
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time, filtered_data, color='black', linewidth=1)
for i, ripple in enumerate(window_ripples):
    start, end = ripple / liset.fs
    ax2.axvspan(start, end, color='yellow', alpha=0.3, label="GT Ripple" if i == 0 else None)
ax2.set_title("B. Ripple Band (100–250 Hz)")
ax2.set_ylabel("Amplitude")

# --- Row 3: ADM (30 kHz) ---
ax3 = fig.add_subplot(gs[2, :])
ax3.vlines(up_spike_times, ymin=0, ymax=1, color='red', linewidth=0.5)
ax3.vlines(down_spike_times, ymin=-1, ymax=0, color='blue', linewidth=0.5)
for i, ripple in enumerate(window_ripples):
    start, end = ripple / liset.fs
    ax3.axvspan(start, end, color='yellow', alpha=0.3, label="GT Ripple" if i == 0 else None)
ax3.set_title("C. ADM Encoding (30 kHz)")
ax3.set_yticks([-0.5, 0.5])
ax3.set_yticklabels(["DN", "UP"])
# ax3.set_ylabel("Spikes")
ax3.set_xlabel("Time [s]")

# --- Row 4 Left: ADM 30kHz zoom ---
ax4 = fig.add_subplot(gs[3, 0])
ax4.vlines(up_spikes_ds, ymin=0, ymax=1, color='red', linewidth=0.5, label="UP (30kHz)")
ax4.vlines(down_spikes_ds, ymin=-1, ymax=0, color='blue', linewidth=0.5, label="DN (30kHz)")
for i, ripple in enumerate(zoom_ripples):
    start, end = ripple / liset.fs
    ax4.axvspan(start, end, color='yellow', alpha=0.3, label="GT Ripple" if i == 0 else None)
ax4.set_title("D1. ADM Encoding (30 kHz, Zoom)")
ax4.set_xlabel("Time [s]")
ax4.set_yticks([-0.5, 0.5])
ax4.set_yticklabels(["DN", "UP"])
ax4.set_xlim(window_downsample)
# ax4.legend()

# --- Row 4 Right: ADM 1kHz zoom ---
ax5 = fig.add_subplot(gs[3, 1])
ax5.vlines(up_spikes_down, ymin=0, ymax=1, color='red', linestyle=":",linewidth=0.5, label="UP (1kHz)")
ax5.vlines(down_spikes_down, ymin=-1, ymax=0, color='blue',linestyle=":", linewidth=0.5, label="DN (1kHz)")
for i, ripple in enumerate(zoom_ripples):
    start, end = ripple / liset.fs
    ax5.axvspan(start, end, color='yellow', alpha=0.3, label="GT Ripple" if i == 0 else None)
ax5.set_title("D2. ADM Encoding (1 kHz, Zoom)")
ax5.set_xlabel("Time [s]")
ax5.set_yticks([-0.5, 0.5])
ax5.set_yticklabels(["DN", "UP"])
ax5.set_xlim(window_downsample)
# ax5.legend()

# --- Final Layout ---
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(curr_dir,"Figure_ADM_Workflow_vertical.png"), dpi=300)
# plt.show()