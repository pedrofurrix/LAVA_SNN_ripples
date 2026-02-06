###########################################################################################
#                                                                                         #
#                   Developed by Pedro Félix Alves
#                      
###########################################################################################

#
#  This module is developed for loading ripple data from Instituto Cajal
#           

# Suppress warnings
import os

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from copy import deepcopy

ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(ROOT_DIR)
if not ROOT_DIR in sys.path:
    sys.path.insert(0, ROOT_DIR)
LISET_DIR=os.path.dirname(os.path.abspath(__file__))
# if not LISET_DIR in sys.path:
#     sys.path.append(LISET_DIR)


from liset_tk.liset_aux import *
from liset_tk.load_data import *
from liset_tk.signal_aid import *
import liset_tk.lists_sessions as lists_sessions
from liset_tk.gt_annotations import get_ripple_events,configs



class liset_tk_extra():
    """
    Class for handling data processing and visualization related to ripple events.
    Parameters:
    - data_path (str): Path to the directory containing the data.
    - shank (int): Shank of the electrode.
    - downsample (bool, optional): Whether to downsample the data. Default is False.
    - normalize (bool, optional): Whether to normalize the data. Default is True.
    - Start (int, optional): Starting sample index. Default is 0. original freq
    - numSamples (int, optional): Number of samples. Default is False. original freq
    - verbose (bool, optional): Whether to display verbose output. Default is True.
    """
     
    def __init__(self, data_path, name, shank=None, downsample = False, normalize = False, numSamples = False, start = 0, verbose=True, original_fs=30000,channels=None,load_data=True,scale_data=False,annotation_type="manual_GT"):
        self.data_path=os.path.join(data_path,name)

        if shank is None:
            shank=lists_sessions.shank_sessions[name]
            
        self.shank=shank
        # Set the verbose
        self.verbose = verbose
        self.numSamples = numSamples
        self.start = start
        self.original_fs = original_fs
        self.scale_data=scale_data
        self.channel_num=32
        self.channels=channels
        self.name=name
        if not self.numSamples:
            self.numSamples = np.inf # read the whole file

        if downsample:
            self.downsampled_fs = downsample
            self.fs_conv_fact = self.original_fs/self.downsampled_fs
            self.fs=self.downsampled_fs
        else:
            self.fs_conv_fact = 1
            self.fs=self.original_fs
        
        self.load_data=load_data
        
        if self.load_data:
            # Load the data.
            self.load(self.data_path, shank, downsample = downsample, normalize=normalize)
        else:
            self.duration=self.get_duration(self.data_path)
        # Try to load the ripples if the path contain a file with ripple times.
        # Only load ripples in the interval selected of data (chunk)
        ripples=self.load_ripples(self.data_path)
        self.ripples_GT = self.ripples_in_chunk(ripples, start, numSamples, self.fs, self.fs_conv_fact)

        self.num_ripples = len(self.ripples_GT)
        self.ripples_GT = (self.ripples_GT * self.fs - start / self.fs_conv_fact).astype(int)
        if self.verbose:
            print(f'Number of ripples in the selected chunk: {self.num_ripples}')

    def get_duration(self,path):
        """
        Get the duration of the data in seconds.

        Parameters:
        - path (str): Path to the directory containing the .dat file.

        Returns:
        - duration (float): Duration of the data in seconds.
        """
        try:
            filename = f"{path}/{[i for i in os.listdir(path) if i.endswith('.dat')][0]}"
            file_len = os.path.getsize(filename=filename)
            self.file_samples = file_len // (self.channel_num * 2)
            duration = self.file_samples / self.original_fs
            return duration
        except:
            if self.verbose:
                print('.dat file not in path')
            return False   

    def ripples_in_chunk(self, ripples, start, numSamples, fs, prop):
        if not numSamples:
            numSamples = self.file_samples - self.start

        in_chunk = ripples[(ripples[:,0] > start/prop/fs) & (ripples[:,0] < (start + numSamples)/prop/fs)]

        return in_chunk

    def get_gt_annotations(self,):
        self.manual_GT=self.ripples_GT
        if self.annotation_type is not None:
            if self.annotation_type=="manual_GT":
                pass
            else:
                
                if hasattr(self, 'data'):
                    channel=lists_sessions.channel_sessions[self.name]-1
                    if self.data.shape[1]>1:
                        channel=self.channels_shank.index(channel)
                else:
                    self.load(self.data_path, shank=None, downsample = False, normalize=self.normalize,invert=False, channels=[channel])
                    channel=0
                info=get_ripple_events(self.data[:,channel],self.fs,config=self.annotation_type)
                #get start and end times from annnotations...
                ripple_idx = info[["start_idx","end_idx"]].to_numpy()
            self.ripples_GT=ripple_idx    
    
    def load_dat(self, path, channels, numSamples = False, verbose=False):
        """
        Load data from a .dat file.

        Parameters:
        - path (str): Path to the directory containing the .dat file.
        - channels (list): List of channel IDs to load.
        - numSamples (int, optional): Number of samples to load. Default is False (load all samples).
        - sampleSize (int, optional): Size of each sample in bytes. Default is 2.
        - verbose (bool, optional): Whether to display verbose output. Default is False.

        Returns:
        - data (numpy.ndarray): Loaded data as a NumPy array.
        """

        try:
            filename = f"{path}/{[i for i in os.listdir(path) if i.endswith('.dat')][0]}"
            self.file_len = os.path.getsize(filename=filename)
            self.file_samples = self.file_len // (self.channel_num * 2)
        except:
            if self.verbose:
                print('.dat file not in path')
            return False
        
        nChannels = len(channels)
        if (len(channels) > nChannels):
            if self.verbose:
                print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
            return False
        
        start = self.start * self.channel_num * 2
        numSamples = self.numSamples * self.channel_num * 2

        if start > self.file_len:
            if self.verbose:
                print(f'the start must be lower than the total file samples.\nTotal file samples: {self.file_samples}')
            return False
        if (numSamples + start) > self.file_len:
            numSamples = self.file_len - start


        if (self.file_len < numSamples) or ((numSamples + self.start) > self.file_len):
            if self.verbose:
                print(f'file has only {self.file_samples} samples')
            return False
            
        with open(filename, "rb") as f:
            # Put the reader at the starting index
            f.seek(start)

            if numSamples:
                raw = f.read(numSamples)
            else:
                raw = f.read(self.file_len - start)
            data = np.frombuffer(raw, dtype=np.int16)
            data = RAW2ORDERED(data, channels,num_channels_raw=self.channel_num)
            return data
            
    def load_ripples(self, path):
        ripple_path=os.path.join(path,'events',"events_selected_manually.txt") 
        if self. verbose:
            print(f'Loading ripples from {ripple_path}')
        try: 
            ripples = np.loadtxt(ripple_path) 
            if len(ripples.shape) == 1: 
                ripples = ripples.reshape(-1, 2) 
            return ripples
        except:
            if self.verbose:
                print('No ripple file found in path')
            raise FileNotFoundError('No ripple file found in path')

    def load(self, data_path, shank, downsample, normalize):
        """
        Load all, optionally downsample and normalize it.

        Parameters:
        - data_path (str): Path to the data directory.
        - shank (int): Shank of the electrode.
        - downsample (float): Downsample factor.
        - normalize (bool): Whether to normalize the data.
    
        Returns:
        - data (numpy.ndarray): Loaded and processed data.
        """
        if self.channels is None:
            self.channels=[24,20,23,27,25,21,22,26,28,16,19,31,29,17,18,30,15,3,0,12,14,2,1,13,11,7,4,8,10,6,5,9]
            self.channels_shank = self.channels[(shank-1)*8 : shank*8]
        else:
            self.channels_shank=self.channels
        
        
        raw_data = self.load_dat(data_path, self.channels_shank, numSamples=self.numSamples)

        if hasattr(raw_data, 'shape'):
            self.data = self.clean(raw_data, downsample, normalize)
            self.duration = self.data.shape[0]/self.fs


    def clean(self, data, downsample, normalize):
        """
        Clean the loaded data by downsampling and normalizing it.

        Parameters:
        - data (numpy.ndarray): Raw data to be cleaned.
        - downsample (bool): Whether to downsample the data.
        - normalize (bool): Whether to normalize the data.

        Returns:
        - data (numpy.ndarray): Cleaned data after downsampling and normalization.
        """

        if downsample:
            self.fs = self.downsampled_fs
            # Downsample data
            if self.verbose:
                print("Downsampling data from %d Hz to %d Hz..."%(self.original_fs, self.downsampled_fs), end=" ")
            data = downsample_data(data, self.original_fs, self.downsampled_fs)
            if self.verbose:
                print("Done!")
        else:
            self.fs = self.original_fs


        if normalize:
            # Normalize it with z-score
            if self.verbose:
                print("Normalizing data...", end=" ")
            data = z_score_normalization(data)
            if self.verbose:
                print("Done!")
                print("Shape of loaded data after downsampling and z-score: ", np.shape(data))

        if self.scale_data and not normalize:
            self.bit_uvolts=0.1949999928
            self.ttl_bit_volts=0.0001525879
            data = data.astype(np.float32)
            data[:,:32] *= (self.bit_uvolts * 1e-6) # Convert to volts
            if self.channel_num==40 and 32 in self.channels:
                data[:,32:] *= self.ttl_bit_volts  # Convert TTL channel to volts
        return data
    
    def plot_visualize(self, ch=None, offset=0, filtered=None, extend=0.5, title='Overview', window=None):
        """
        Plot data with overlays for:
        - Ground truth ripples (yellow)
        - Model predicted ripples (blue)
        - Light stimulation TTLs (red)

        Parameters
        ----------
        ch : int or list of ints, optional
            Channel(s) to plot. If None, plots the first channel.
        offset : float, optional
            Vertical offset between channels.
        filtered : tuple (low, high), optional
            Bandpass filter range (Hz).
        extend : float, optional
            Extra context (in seconds) around each event.
        title : str, optional
            Title for the plot.
        window : tuple (start, end), optional
            Time window in seconds to plot (e.g., (10, 20)).
        """

        # ---------------------
        # Select channels
        # ---------------------
        if ch is None:
            ch = [0]
        elif isinstance(ch, int):
            ch = [ch]

        n_samples = self.data.shape[0]
        time = np.arange(n_samples) / self.fs

        # ---------------------
        # Apply time window
        # ---------------------
        if window is not None:
            start_s, end_s = window
            start_idx = int(start_s * self.fs)
            end_idx = int(end_s * self.fs)
            time = time[start_idx:end_idx]
            data_slice = self.data[start_idx:end_idx, :]
        else:
            data_slice = self.data

        fig, ax = plt.subplots(figsize=(15, 6))
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized units)")

        # ---------------------
        # Plot data (optionally filtered)
        # ---------------------
        for i, ch_idx in enumerate(ch):
            sig = deepcopy(data_slice[:, ch_idx])
            if filtered:
                from signal_aid import bandpass_filter
                sig = bandpass_filter(sig, filtered, self.fs)

            ax.plot(time, sig + i * offset, color="black", lw=0.8, label=f"Ch {ch_idx}" if i == 0 else "")

        min_y, max_y = ax.get_ylim()

        # ---------------------
        # Helper function to plot intervals safely within window
        # ---------------------
        def plot_intervals(intervals, color, label):
            if window is not None:
                mask = (intervals[:, 1] / self.fs > start_s) & (intervals[:, 0] / self.fs < end_s)
                intervals = intervals[mask]
            for i, r in enumerate(intervals):
                ax.fill_between(r / self.fs, min_y, max_y, color=color, alpha=0.3, label=label if i == 0 else "")

        # ---------------------
        # Ground truth ripples (yellow)
        # ---------------------
        if hasattr(self, "ripples_GT") and self.ripples_GT is not None:
            plot_intervals(self.ripples_GT, "yellow", "Ground truth")
        # ---------------------
        # Beautify
        # ---------------------
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    data_path=r"C:\PedroFelix\extra_data\original_data"
    name="Calbai32FPGA_251003_144832"
    data=liset_tk_extra(data_path,name,shank=None,downsample=4000,normalize=True,start=0,numSamples=False,verbose=True)
    data.plot_visualize(ch=[0],offset=5,filtered=(100,250),extend=0.5,title='Overview',window=None)