###########################################################################################
#                                                                                         #
#                         Developed by: Pedro Félix Alves                                 #
#                           Contact: pedrofelixalves@gmail.com                            #
#                                                                                         #
###########################################################################################

#
#  This module is developed for 3 main reasons:
#           - Loading correctly the data obtained from the Open Ephys system.
#           - Visualizing performance and data quality.
#           - Assisting in the analysis of ripple events.

import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from copy import deepcopy
from liset_tk.liset_aux import *
from liset_tk.load_data import *
from liset_tk.signal_aid import *
import pickle as pkl
from datetime import datetime

try:
    from liset_tk.gt_annotations import *
    import liset_tk.lists_sessions as lists_sessions
except (ImportError, ModuleNotFoundError):
    import lists_sessions as lists_sessions
    from gt_annotations import *

class read_data():
    """
    Class for handling data processing and visualization related to ripple events.

    Parameters:
    - data_path (str): Path to the directory containing the data.
    - shank (int): Shank of the electrode.
    - downsample (bool, optional): Whether to downsample the data. Default is False.
    - normalize (bool, optional): Whether to normalize the data. Default is True.
    - numSamples (int, optional): Number of samples. Default is False.
    - verbose (bool, optional): Whether to display verbose output. Default is True.
    """
    # TODO: There are still some problems - namely the annotation does not obey to 
    def __init__(self, data_path, name, shank=0, downsample = False, normalize = True, numSamples = False, start = 0, verbose=True, invert=False,offset=0.16,buffer_size=20,load_data=True, channels=None,scale_data=False,recording_node=1,annotation_type=None,remove_artifacts=False):
        
        self.oe_path=os.path.join(data_path,"Open Ephys",name)
        self.annotation_path=os.path.join(data_path,"annotations")
        self.offline_path=os.path.join(data_path,"offline",f"{buffer_size}")
        
        self.scale_data=scale_data
        # Set the verbose
        self.verbose = verbose
        if self.verbose:
            print(f"Loading data from session: {name}")
        self.numSamples = numSamples
        self.start = start
        self.load_data=load_data
        self.annotated=RippleEvents(offset=offset)
        self.from_data=RippleEvents(offset=offset)
        self.normalize=normalize
        self.name=name
        self.channels=channels
        self.recording_node=recording_node
        self.annotation_type=annotation_type
        self.remove_artifacts=remove_artifacts
        self.get_channel_num(self.oe_path)


        if downsample:
            self.downsampled_fs = downsample
            self.fs_conv_fact = self.original_fs/self.downsampled_fs
        else:
            self.fs_conv_fact = 1
            self.downsampled_fs = self.original_fs

 

        # Load the data
        if self.load_data:
            self.load(self.oe_path, shank, downsample = downsample, normalize=normalize,invert=invert, channels=channels)
            self.artifact_removal()
        else:
            self.load_only_ripples(self.oe_path,invert=invert)
            self.duration=self.get_duration(self.oe_path)    
        
        self.get_gt_annotations() # so that it's already done with the artifact removal (if needed)

        self._check_data()
        if self.verbose:
            print(f"✅ Loading complete!- session {name}")

    def get_duration(self,path):
        """
        Get the duration of the data in seconds.

        Parameters:
        - path (str): Path to the directory containing the .dat file.

        Returns:
        - duration (float): Duration of the data in seconds.
        """
        try:
            list_dir=os.listdir(path)
            file=list_dir[self.recording_node] # Record Node 0 or 1 - #TODO: make it more robust and see if it isn't better to have 0...
            record_file=os.path.join(path,file,"experiment1","recording1","continuous")
            list_2=os.listdir(record_file)
            folder=os.path.join(record_file,list_2[0])
            filename=os.path.join(folder,"continuous.dat")
            self.file_len = os.path.getsize(filename=filename)
            self.file_samples = self.file_len // (self.channel_num * 2)
            duration = self.file_samples / self.original_fs
            return duration
        except:
            if self.verbose:
                print('.dat file not in path')
            return False          

    def load_only_ripples(self,path,invert=False):
        try:
            list_dir=os.listdir(path)
            file=list_dir[1]
            record_file=os.path.join(path,file,"experiment1","recording1","continuous")
            list_2=os.listdir(record_file)
            folder=os.path.join(record_file,list_2[0])
            timestamps_file= os.path.join(folder,"timestamps.npy")
            self.timestamps = np.load(timestamps_file)
        except:
            if self.verbose:
                print('❗timestamps.npy file not in path❗, cannot load ripples.')
            return False
        self.numSamples=np.inf
        self.fs = self.downsampled_fs # Does it work when downsampling?
        
        self.load_annotations(self.annotation_path,self.name)
        self.load_ripple_times(self.oe_path,invert)
        self.load_offline()
        # self.get_gt_annotations()
        
    def ripples_in_chunk(self, ripples, start, numSamples, fs, prop):
        if not numSamples:
            numSamples = self.file_samples - self.start

        in_chunk = ripples[(ripples[:,0] > start/prop/fs) & (ripples[:,0] < (start + numSamples)/prop/fs)]

        return in_chunk

    def get_gt_annotations(self,):
        self.manual_GT=self.annotated.ripples_GT
        if self.annotation_type is not None:
            if self.annotation_type=="manual_GT":
                return self.annotated.ripples_GT # Samples
            else:
                if hasattr(self, 'data'):
                    
                    channel_og=lists_sessions.channel_sessions[self.name]-1
                    channel = np.where(np.array(self.channels) == channel_og)[0][0]
                else:
                    self.load(self.oe_path, shank=0, downsample = self.downsampled_fs, normalize=self.normalize,invert=False, channels=[lists_sessions.channel_sessions[self.name]-1])
                    channel=0
                info=get_ripple_events(self.data[:,channel],self.fs,config=self.annotation_type)
                #get start and end times from annnotations...
                ripple_idx = info[["start_idx","end_idx"]].to_numpy()
                self.annotated.ripples_GT=ripple_idx    

    def load_dat(self, path, channels, numSamples = False, verbose=False):
        """
        Load data from a .dat file.

        Parameters:
        - path (str): Path to the directory containing the .dat file.
        - channels (list): Lis  t of channel IDs to load.
        - numSamples (int, optional): Number of samples to load. Default is False (load all samples).
        - sampleSize (int, optional): Size of each sample in bytes. Default is 2.
        - verbose (bool, optional): Whether to display verbose output. Default is False.

        Returns:
        - data (numpy.ndarray): Loaded data as a NumPy array.
        """
        try:
            list_dir=os.listdir(path)
            file=list_dir[self.recording_node] # Record Node 0 or 1 - #TODO: make it more robust and see if it isn't better to have 0...
            record_file=os.path.join(path,file,"experiment1","recording1","continuous")
            list_2=os.listdir(record_file)
            folder=os.path.join(record_file,list_2[0])
            filename=os.path.join(folder,"continuous.dat")
            self.file_len = os.path.getsize(filename=filename)
            self.file_samples = self.file_len // (self.channel_num * 2)
            timestamps_file= os.path.join(folder,"timestamps.npy")
            self.timestamps = np.load(timestamps_file)
            if self.timestamps[0]>0:
                print(f"Warning: The first timestamp is not zero, there might be an offset in the data - {self.timestamps[0]}")
            else:
                print("Starting from 0 :D")
        except:
            if self.verbose:
                print('❗timestamps.npy file not in path❗, cannot load ripples/data.')
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
            data = RAW2ORDERED(data, channels,num_channels_raw=self.channel_num )
            return data
            
    def get_channel_num(self,data_path):
        list_dir=os.listdir(data_path)
        for file in list_dir:
            if "Record Node" in file:
                try:
                    record_file=os.path.join(data_path,file,"experiment1","recording1","structure.oebin")
                    with open(record_file, "r") as f:
                        data = json.load(f)
                        num_channels = data["continuous"][0]["num_channels"]
                        self.original_fs=data["continuous"][0]["sample_rate"]
                        self.channel_num=num_channels
                        self.bit_uvolts=data["continuous"][0]["channels"][0]["bit_volts"] # conversion to microvolts...
                        for channel in data["continuous"][0]["channels"]:
                            if "ADC" in channel["channel_name"]:
                                self.ttl_bit_volts=channel["bit_volts"]
                                break

                except Exception as e:
                    print(f"Error reading structure.oebin: {e}")
                    self.channel_num=32  # default value
        if self.verbose:
            print(f"Loaded {num_channels} channels, with a original frequency of {self.original_fs} Hz")

    def artifact_removal(self):
        channel_og=lists_sessions.channel_sessions[self.name]-1
        channel = np.where(np.array(self.channels) == channel_og)[0][0]
        if self.remove_artifacts:
            if hasattr(self, "data"):
                if not hasattr(self.from_data, "light_stim") or self.from_data.light_stim is None:
                    print("⚠️ No light stimulation events loaded, cannot perform artifact removal.")
                    return
                stim_intervals = self.from_data.light_stim # samples

                fs = self.fs
                clean_data = self.data.copy().astype(np.float32)
                buffer_samp = int(self.remove_artifacts* fs / 1000)
                # Baseline check window
                base_ms = 10.0
                base_samp = int(base_ms * 1e-3 * fs)
                for s, e in stim_intervals:
                    # Safety check for indices
                    if s - base_samp < 0 or e + base_samp >= len(clean_data):
                        continue
                    
                    # --- 1. Level Alignment (remove DC shift) ---
                    # Get median baseline just before and after the stim
                    pre_base = clean_data[s - base_samp : s - buffer_samp, channel]
                    post_base = clean_data[e + buffer_samp : e + base_samp, channel]
                    
                    # Get median level during stim (ignoring edges)
                    if (e - buffer_samp) > (s + buffer_samp):
                        stim_segment = clean_data[s + buffer_samp : e - buffer_samp, channel]
                    else:
                        # Stim too short for buffer, take center
                        mid = (s + e) // 2
                        stim_segment = clean_data[mid : mid+1, channel]

                    target_level = (np.median(pre_base) + np.median(post_base)) / 2
                    current_level = np.median(stim_segment)
                    
                    offset = current_level - target_level
                    
                    # Subtract the offset from the continuous block
                    clean_data[s:e, channel] -= offset

                    # --- 2. Linear Interpolation of Edges (Kill transients) ---
                    # Interpolate across the ON transition [s - buffer, s + buffer]
                    p_start = s - buffer_samp
                    p_end   = s + buffer_samp
                    if p_start >= 0 and p_end < len(clean_data):
                        y_start = clean_data[p_start, channel]
                        y_end   = clean_data[p_end, channel]
                        clean_data[p_start:p_end, channel] = np.linspace(y_start, y_end, p_end - p_start)

                    # Interpolate across the OFF transition [e - buffer, e + buffer]
                    p_start = e - buffer_samp
                    p_end   = e + buffer_samp
                    if p_start >= 0 and p_end < len(clean_data):
                        y_start = clean_data[p_start, channel] # Note: clean_data[p_start] already shifted above
                        y_end   = clean_data[p_end, channel]   # clean_data[p_end] is outside stim, unshifted
                        clean_data[p_start:p_end, channel] = np.linspace(y_start, y_end, p_end - p_start)
                self.data = clean_data


    def load(self, data_path, shank, downsample, normalize,invert, channels=None):
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

        # try:
        #     info = loadmat(f'{data_path}/info.mat')
        # except:
        #     try:
        #         info = loadmat(f'{data_path}/neurospark.mat')
        #     except:
        #         print('.mat file cannot be opened or is not in path.')
        #         return
            
        # try:
        #     channels = info['neurosparkmat']['channels'][0][0][8 * (shank -1):8 * shank]
        # except Exception as err:
        #     print(f'No data available for shank {shank}\n\n{err}')
        #     return 
        # channels=[24,20,23,27,25,21,22,26,28,16,19,31,29,17,18,30,15,3,0,12,14,2,1,13,11,7,4,8,10,6,5,9]
        if channels is None:
            channels=range(self.channel_num)
            self.channels=channels
        else:
            if self.remove_artifacts and self.name in lists_sessions.light_on:
                self.channels.append(32) if 32 not in self.channels else None
        # channels=channels[shank*8-8:shank*8]
      
        raw_data = self.load_dat(data_path, self.channels, numSamples=self.numSamples)

        if hasattr(raw_data, 'shape'):
            self.data = self.clean(raw_data, downsample, normalize)
            self.duration = self.data.shape[0]/self.fs
        if self.load_data:
            self.load_annotations(self.annotation_path,self.name)
            self.load_ripple_times(data_path,invert)
            self.load_offline()
            

    


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
            data = data.astype(np.float32)
            data[:,:32] *= (self.bit_uvolts * 1e-6) # Convert to volts
            if self.channel_num==40 and 32 in self.channels:
                data[:,32:] *= self.ttl_bit_volts  # Convert TTL channel to volts
        return data

    def load_ripple_times(self,path,invert=False):
        
        list_dir=os.listdir(path)
        file=list_dir[1]
        record_file=os.path.join(path,file,"experiment1","recording1", "events")
        for i in os.listdir(record_file):
            if "Network_Events" in i:
                event_folder=os.path.join(record_file,i,"TTL")

        sample_numbers = os.path.join(event_folder, "sample_numbers.npy")
        timestamps_ttl = os.path.join(event_folder, "timestamps.npy")

        sample_numbers = np.load(sample_numbers)   # might not be used here
        # if self.timestamps[0]>0:
        #     self.timestamps -= self.timestamps[0]
        # self.duration=self.timestamps[-1]
        
        timestamps_ttl = np.load(timestamps_ttl) - self.timestamps[0]  # Adjust timestamps to start from zero
        if invert:
            detections=timestamps_ttl[1:]
        else:
            detections=timestamps_ttl[0:]
        # Ensure even length (pairs of start/end)
        if len(detections) % 2 != 0:
            detections=np.append(detections, detections[-1]+0.020)  # or handle as appropriate
            # raise ValueError("Timestamps array must contain an even number of elements (start/end pairs).")

        ripples = detections.reshape(-1, 2)  # shape [X, 2]
        self.from_data.snn_predicts=self.ripples_in_chunk(ripples,self.start,self.numSamples,self.fs,self.fs_conv_fact)
        self.get_ttls(invert)
        self.from_data.seconds_to_samples(self.fs, self.start, self.fs_conv_fact)

    
    def get_ttls(self, invert=False):
        if self.normalize:
            min_max=0
        else:
            min_max=0
        
        if self.timestamps[0]>0:
            self.timestamps -= self.timestamps[0]
        self.duration=self.timestamps[-1]

        if 32 in self.channels:
            if hasattr(self, 'data'):
                ttl_channel = np.where(np.array(self.channels) == 32)[0][0]
                ttl_signal = self.data[:, ttl_channel]
            else:
                print("⚠️ Data not loaded, cannot extract TTL signal.")
                return

            if ttl_signal.max() <min_max or self.annotated.light_stim is None:
                print("⚠️ TTL signal not detected or too low amplitude.")
                return

            else:

                # --- Step 1: Set a threshold automatically (midpoint between min and max)
                threshold = (ttl_signal.max() + ttl_signal.min()) / 2

                # --- Step 2: Detect rising edges (transitions from low -> high)
                ttl_binary = ttl_signal > threshold # detects when high
                edges = np.diff(ttl_binary.astype(int)) # detects transitions - low to high

                # Rising edges (0 -> 1)
                rising_edges = np.where(edges == 1)[0]

                # Falling edges (1 -> 0)
                falling_edges = np.where(edges == -1)[0]


                if len(falling_edges)!=len(rising_edges):
                    if len(rising_edges)>len(falling_edges):
                        print(" ⚠️ More falling edges than rising edges, adjusting...")
                        falling_edges = np.append(falling_edges, rising_edges[-1] + int(0.001 * self.fs))
                        
                    else:    
                        print(" ⚠️ More rising edges than falling edges, adjusting...")
                        min_len=min(len(falling_edges),len(rising_edges))
                        # print(min_len)
                        rising_edges=rising_edges[0:min_len]
                        falling_edges=falling_edges[1:min_len+1]
                
                ttls= np.column_stack((rising_edges, falling_edges))
                ttl_times=ttls/self.fs
                self.from_data.light_stim=self.ripples_in_chunk(ttl_times,self.start,self.numSamples,self.fs,self.fs_conv_fact)
    
    def _check_data(self):
        if self.annotated.ripples_GT is None:
            print("⚠️ No ground truth annotations loaded.")
        else:
            print(f"Loaded {len(self.annotated.ripples_GT)} ground truth ripple events.")
        if self.annotated.snn_predicts is None:
            print("⚠️ No ground truth SNN predictions loaded.")
        elif self.from_data.snn_predicts.shape != self.annotated.snn_predicts.shape:
            print(f"⚠️ Warning: Number of SNN predictions from data ({self.from_data.snn_predicts.shape[0]}) does not match number of annotations ({self.annotated.snn_predicts.shape[0]}).")
        else:
            print(f"Loaded {len(self.from_data.snn_predicts)} SNN predicted ripple events.")

        if self.from_data.light_stim is None and self.annotated.light_stim is None:
            print("⚠️ No light stimulation events loaded.")
        else:
            if self.annotated.light_stim is None:
                print("⚠️ No ground truth light stimulation annotations loaded.")
                print(f"Loaded {len(self.from_data.light_stim)} light stimulation events.")
            elif self.from_data.light_stim is None:
                print("⚠️ No light stimulation events loaded from data.")
            elif self.from_data.light_stim.shape != self.annotated.light_stim.shape:
                print(f"⚠️ Warning: Number of raw light stimulation events ({self.from_data.light_stim.shape[0]}) does not match number of annotated light stimulation events ({self.annotated.light_stim.shape[0]}).")
            else:
                print(f"Loaded {len(self.from_data.light_stim)} light stimulation events.")


    def load_annotations(self,path,name):
        list_dir=os.listdir(path)
        for file in list_dir:
            sessions=os.listdir(os.path.join(path,file))
            # print(sessions)
            if any(name in s for s in sessions):
                folder_path=os.path.join(path,file,f"events_{name}","events")
                print(folder_path)
                self.annotated.ripples_GT=self.ripples_in_chunk(np.loadtxt(os.path.join(folder_path,"events_selected_manually.txt")),self.start,self.numSamples,self.fs,self.fs_conv_fact)

                self.annotated.snn_predicts=self.ripples_in_chunk(np.loadtxt(os.path.join(folder_path,"events_SNN.txt")),self.start,self.numSamples,self.fs,self.fs_conv_fact)
                # print(self.annotated.snn_predicts)
                try:
                    self.annotated.light_stim=self.ripples_in_chunk(np.loadtxt(os.path.join(folder_path,"events_light.txt")),self.start,self.numSamples,self.fs,self.fs_conv_fact)
                    print(f"Loaded {len(self.annotated.light_stim)} annotated light stimulation events.")
                except Exception as e:
                    print(f"Light stim file not found: {e}")
                break
            else:
                print(f"Annotation for session {name} not found in {os.path.join(path,file)}.")
        self.annotated.seconds_to_samples(self.fs, self.start, self.fs_conv_fact)
    
    def load_offline(self,):
        filename=f"spike_data_offline_{self.name}.pkl"
        try:
            with open(os.path.join(self.offline_path,filename), 'rb') as f:
                offline_data = pkl.load(f)
            self.list_TTLs = offline_data.get("ttls", [[], [], []])  # [Detection, Input, Seizure]
            self.offline_detections = np.round(np.array(self.list_TTLs[0]) / self.fs_conv_fact).astype(int)            
        except Exception as e:
            print(f"Error loading offline data: {e}")
            self.offline_detections = np.array([])

    def plot_offline(self, ch=None, offset=0, filtered=None, extend=0.5, title='Overview', window=None):
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
        if hasattr(self.annotated, "ripples_GT") and self.annotated.ripples_GT is not None:
            plot_intervals(self.annotated.ripples_GT, "yellow", "Ground truth")

        # ---------------------
        # Predicted ripples (blue)
        # ---------------------
        # if hasattr(self.from_data, "snn_predicts") and self.from_data.snn_predicts is not None:
        #     # plot_intervals(self.from_data.snn_predicts, "tab:blue", "Predicted")
        #     starts = self.from_data.snn_predicts[:, 0] / self.fs
        #     if window is not None:
        #         starts = starts[(starts >= start_s) & (starts <= end_s)]
        #     ax.vlines(starts, min_y, max_y, color="tab:blue", alpha=0.5, lw=0.5,label="Predicted")

        # if hasattr(self.annotated, "snn_predicts") and self.annotated.snn_predicts is not None:
        #     starts = self.annotated.snn_predicts[:, 0] / self.fs
        #     if window is not None:
        #         starts = starts[(starts >= start_s) & (starts <= end_s)]
        #     ax.vlines(starts, min_y, max_y, color="green", alpha=0.5, lw=0.5,label="Annotated Predicted")

        # ---------------------
        # Light stimulation (red)
        # ---------------------
        if hasattr(self.from_data, "light_stim") and self.from_data.light_stim is not None:
            plot_intervals(self.from_data.light_stim, "red", "Light stim")
            starts = self.from_data.light_stim[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="red", alpha=0.5, lw=0.5)

        if hasattr(self.annotated, "light_stim") and self.annotated.light_stim is not None:
            plot_intervals(self.annotated.light_stim, "orange", "Annotated light stim")
            starts = self.annotated.light_stim[:, 0] / self.fs
            if window is not None:
                starts = starts[(starts >= start_s) & (starts <= end_s)]
            ax.vlines(starts, min_y, max_y, color="orange", alpha=0.5, lw=0.5)

        # ---------------------
        # Offline detections (purple dashed)
        # ---------------------
        if hasattr(self, "offline_detections") and self.offline_detections.size > 0:
            offline_starts = self.offline_detections / self.fs
            if window is not None:
                offline_starts = offline_starts[(offline_starts >= start_s) & (offline_starts <= end_s)]
            ax.vlines(offline_starts, min_y, max_y, color="purple", alpha=0.5, lw=0.5, ls='--', label="Offline detections")
        # ---------------------
        # Beautify
        # ---------------------
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class RippleEvents():
    def __init__(self,offset):
        self.ripples_GT = None
        self.snn_predicts = None
        self.light_stim = None
        self.offset=offset

    def seconds_to_samples(self, fs, start, fs_conv_fact):
        """
        Convert event times from seconds to sample indices.

        Parameters
        ----------
        fs : float
            Sampling frequency (Hz).
        start : int
            Start sample index of the chunk.
        fs_conv_fact : float
            Conversion factor if data were downsampled.
        """
        def convert(arr):
            return (arr * fs - start / fs_conv_fact).astype(int)

        if isinstance(self.ripples_GT, np.ndarray) and self.ripples_GT.size > 0:
            self.ripples_GT = convert(self.ripples_GT)
        if isinstance(self.snn_predicts, np.ndarray) and self.snn_predicts.size > 0:
            self.snn_predicts = convert(self.snn_predicts-self.offset)
        if isinstance(self.light_stim, np.ndarray) and self.light_stim.size > 0:
            self.light_stim = convert(self.light_stim)


# data_path=r"D:\Madrid_tests"
# name="2025-09-24_14-37-17"

# invert=True

# liset=read_data(data_path, name, downsample=1000, normalize=True, invert=invert,offset=0)
# # print(liset.offline_detections[:10]/liset.fs)
# liset.plot_all(ch=[20,],offset=0.3,filtered=(150,250))
# # dic=plot_difference(liset)

# liset.plot_offline(ch=[20,],offset=0.3,)