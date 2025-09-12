from liset_seizures import liset_seizures
import os
import matplotlib.pyplot as plt
import numpy as np
from liset_tk.signal_aid import bandpass_filter

parent=r"E:\neurospark_mat\KA MODEL TRANSITION SESSIONS"
# parent=r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\KA MODEL TRANSITION SESSIONS"
sessions=os.listdir(parent)

for i,s in enumerate(sessions):
    print(f"{i}: {s}")

idx = int(input("Select session index: "))
session = sessions[idx]
data_path=os.path.join(parent,session)
print(f"Selected: {idx}: {session}")

liset=liset_seizures(data_path,shank=3,downsample=1000,normalize=True,start=0,verbose=True,)#numSamples=1500*30000)

# --- Plotting the Whole Signal for a Single Channel ---

# 1. Define which single channel you want to plot
channel_to_plot = 3
print(f"Plotting the full signal for channel: {channel_to_plot}")

# 2. Select the data for that single channel from the entire loaded signal
data_channel = liset.data[:, channel_to_plot]
print(data_channel[-20:])
print(f"Total data shape: {data_channel.shape}")
print(f"Data duration (s): {data_channel.shape[0] / liset.fs}")
# 3. Apply the bandpass filter to the entire channel data
filtered = bandpass_filter(data_channel, [1, 70], liset.fs, order=4)

# 4. Create a time vector for the entire duration of the signal
time = np.arange(data_channel.shape[0]) / liset.fs

# Get min/max values for plotting vertical lines and fills
min_val, max_val = np.min(data_channel), np.max(data_channel)

# 5. Create the figure with two subplots (raw and filtered)
fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

# Plot Raw Signal
axes[0].plot(time, data_channel, label="Raw Signal")
axes[0].set_title(f"Full Signal - Channel {channel_to_plot}")
axes[0].set_ylabel("Amplitude (Z-score)")
axes[0].grid(True)

# Plot Filtered Signal
axes[1].plot(time, filtered, label="Filtered Signal (0.1-70 Hz)", color='orange')
axes[1].set_title(f"Filtered Signal - Channel {channel_to_plot}")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)


# --- Overlay Events on BOTH Subplots ---

# Plot IISs (if they exist)
if hasattr(liset, "IISs_times"):
    IIS_times_sec = liset.IISs_times / liset.fs
    print(f"Found {len(IIS_times_sec)} IIS events.")
    for ax in axes:
        # Plot all IISs as vertical lines
        ax.vlines(IIS_times_sec, ymin=min_val, ymax=max_val, color="red", alpha=0.5, label="IISs")

# Plot Seizures (if they exist)
if hasattr(liset, "seizure_times"):
    seizure_times_sec = liset.seizure_times / liset.fs
    print(f"Found {seizure_times_sec.shape[0]} seizure events.")
    for i, (start, end) in enumerate(seizure_times_sec):
        for ax in axes:
            # Plot all seizures as filled regions
            fill_SZ = ax.fill_between([start, end], min_val, max_val,
                                      color="tab:purple", alpha=0.3,
                                      # Only add the label once to avoid clutter in the legend
                                      label="Seizure" if i == 0 else None)

# Add legends to each subplot
axes[0].legend()
axes[1].legend()

plt.tight_layout()  # Adjusts plot to prevent labels overlapping
plt.show()