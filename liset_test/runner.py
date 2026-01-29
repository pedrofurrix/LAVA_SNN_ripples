import sys
import os
import pickle as pkl
curr_dir=os.path.dirname(os.path.abspath(__file__))
par_dir=os.path.dirname(curr_dir)
sys.path.insert(0, par_dir)

from liset_tk.load_data import *
from liset_tk.read_data import *
from liset_tk.liset_tk_extra import liset_tk_extra
import liset_tk.lists_sessions as lists_sessions
from liset_test.process_signal import *
from liset_tk.format_predictions import *

import tensorflow.keras.backend as K
import tensorflow.keras as kr

def run_detection_cnn(threshold,path,sessions,channels_sessions):
    curr_dir=os.path.dirname(os.path.abspath(__file__))
    overlapping = True
    window_size = 0.0128
    downsample=1250
    
    save_path=os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(save_path, "cnn_detections",)
    pkl_path = os.path.join(out_dir, f"detections_thr_{threshold}.pkl")

    if os.path.exists(pkl_path):
        print(f"Loading existing predictions from {pkl_path}")
        with open(pkl_path, "rb") as f:
            all_predictions = pkl.load(f)
    else:
        all_predictions = {}

    for session in sessions:
        if session in all_predictions:
            print(f"Session {session} already processed. Skipping.")
            continue
        channel=channels_sessions.get(session,None)-1
        if session in lists_sessions.extra_sessions:
            print(f"Skipping extra session {session}.")
            data,_=load_experimental_data(path,session,downsample=downsample,normalize=True,channel=None,data_reader=liset_tk_extra) # None to load all channels from the shank
        else:
            shank=channel//8
            channels=np.arange(shank*8,shank*8+8)
            data,_=load_experimental_data(path,session,downsample=downsample,normalize=True,channel=channels,data_reader=read_data)
        print("Generating windows...", end=" ")
        if overlapping:    
            stride = 0.0064
            # Separate the data into 12.8ms windows with 6.4ms overlapping
            X = generate_overlapping_windows(data, window_size, stride, downsample)
        else:
            stride = window_size
            X = np.expand_dims(data, 0)
        print("Done!")


        print("Loading CNN model...", end=" ")
        optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        model = kr.models.load_model(os.path.join(curr_dir, "model"), compile=False)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        print("Done!")

        print("Detecting ripples...", end=" ")
        predictions = model.predict(X, verbose=True)
        print("Done!")
        print("Getting detected ripples indexes and times...", end=" ")
        pred_indexes = get_predictions_indexes(data, predictions, window_size=window_size, stride=stride, fs=downsample, threshold=threshold)

        pred_times = pred_indexes / downsample
        print("Done!")
        all_predictions[session] = {
            "pred_indexes": pred_indexes,
            "pred_times": pred_times,
        }
        save_predictions(all_predictions, threshold)

def save_predictions(detections_dict: dict, threshold: float):
    save_path=os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(save_path, "cnn_detections",)
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"detections_thr_{threshold}.pkl")
    with open(pkl_path, "wb") as f:
        pkl.dump(detections_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

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
    #     "2025-09-24_11-34-51",
    #     "2025-09-25_11-21-53",
    #     "2025-09-25_12-52-22",})
    

    # sessions=["2025-09-23_16-17-52"]

    sessions=lists_sessions.extra_sessions
    channel_sessions=lists_sessions.channel_sessions
    for threshold in [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        run_detection_cnn(threshold,path,sessions,channel_sessions)