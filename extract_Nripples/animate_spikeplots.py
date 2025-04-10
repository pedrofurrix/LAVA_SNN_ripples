import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

def animate_level_crossing_progressive(signal, original, sample_num=0, fs=4000, duration=None,fps=5):
    original = original[sample_num, :]
    signal = signal[sample_num, :, :]
    time = np.arange(len(original)) / fs
    
    duration = duration or len(original)/fs  # Use full signal if duration not specified
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plt.suptitle("Progressive Signal & Spikes", y=0.98)
    
    # Dynamic signal line (no static background)
    line, = ax1.plot([], [], 'b-', lw=1)
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(np.min(original)-0.5, np.max(original)+0.5)
    ax1.grid(True)
    
    # Spike trains
    pos_line = ax2.vlines([], [], [], color='r', lw=1.5, label='Positive')
    neg_line = ax2.vlines([], [], [], color='b', lw=1.5, label='Negative')
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0.25, 1.25])
    ax2.set_yticklabels(["Negative", "Positive"])
    ax2.legend()
    ax2.grid(True)
    
    # Set fixed window bounds
    ax1.set_xlim(0, duration)
    ax2.set_xlim(0, duration)
    
    # Pre-compute spike times
    pos_times = time[signal[0] > 0.5]
    neg_times = time[signal[1] > 0.5]
    
    def init():
        line.set_data([], [])
        pos_line.set_segments([])
        neg_line.set_segments([])
        return line, pos_line, neg_line
    
    def update(frame_time):
        # Update signal up to current time
        visible_signal = time <= frame_time
        line.set_data(time[visible_signal], original[visible_signal])
        
        # Update spikes up to current time
        visible_pos = pos_times[pos_times <= frame_time]
        visible_neg = neg_times[neg_times <= frame_time]
        
        pos_segs = np.array([[[t, 1], [t, 1.2]] for t in visible_pos])
        neg_segs = np.array([[[t, 0], [t, 0.2]] for t in visible_neg])
        
        pos_line.set_segments(pos_segs if len(pos_segs) > 0 else [])
        neg_line.set_segments(neg_segs if len(neg_segs) > 0 else [])
        
        return line, pos_line, neg_line
    
    frame_times = np.linspace(0, duration, int(duration*fs/fps))  # 20 fps
    
    ani = animation.FuncAnimation(
        fig, update,
        frames=frame_times,
        init_func=init,
        blit=True,
        interval=50,
        repeat=False
    )
    
    plt.tight_layout()
    return ani

def animate_marcos_progressive(signal, original, sample_num=7, fs=4000, duration=None,fps=5):
    original = original[sample_num, :]
    signal = signal[sample_num, :, :]
    time = np.arange(len(original)) / fs
    
    duration = duration or len(original)/fs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Progressive signal
    line, = ax1.plot([], [], 'steelblue', lw=1)
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(np.min(original)-0.5, np.max(original)+0.5)
    ax1.grid(True)
    
    # Progressive spikes
    scatter = ax2.scatter([], [], s=30, c='r', marker='|')
    ax2.set_ylim(-0.5, signal.shape[0]-0.5)
    ax2.invert_yaxis()
    ax2.set_ylabel("Channel")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)
    
    # Fixed window bounds
    for ax in [ax1, ax2]:
        ax.set_xlim(0, duration)
    
    # Get all spike coordinates
    y_coords, x_coords = np.where(signal > 0.5)
    spike_times = x_coords / fs
    
    def init():
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        return line, scatter
    
    def update(frame_time):
        # Update signal
        visible_signal = time <= frame_time
        line.set_data(time[visible_signal], original[visible_signal])
        
        # Update spikes
        visible_spikes = spike_times <= frame_time
        scatter.set_offsets(np.column_stack([
            spike_times[visible_spikes], 
            y_coords[visible_spikes]
        ]))
        
        return line, scatter
    
    frame_times = np.linspace(0, duration, int(duration*fs/fps))
    
    ani = animation.FuncAnimation(
        fig, update,
        frames=frame_times,
        init_func=init,
        blit=True,
        interval=50,
        repeat=False
    )
    
    plt.tight_layout()
    return ani