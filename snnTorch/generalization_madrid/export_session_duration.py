import os
import pickle
import numpy as np
import pandas as pd
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
# Add parent directories to path to import project modules
curr_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR= os.path.abspath(os.path.join(curr_dir, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from snnTorch.generalization_madrid.process_signal import load_experimental_data
from liset_data_reader.read_data import read_data
from liset_data_reader.liset_tk_extra import liset_tk_extra
import liset_data_reader.lists_sessions as lists_sessions


def process_live_results(
    session,
    data_path_original,
    data_path_extra,
):

    
    # Iterate over sessions
    
        
        
    if session in lists_sessions.extra_sessions: 
        # Load GT ripples
        # We use load_experimental_data to get ripples. 
        # Note: This loads the signal too, which might be slow.
        try:
            # Suppress stdout from load_experimental_data if verbose
            # sys.stdout = open(os.devnull, 'w') 
            _, ripples,duration_s = load_experimental_data(
                data_path_extra,
                session,
                channel=1,
                load_data=False, 
                verbose=False,
                data_reader=liset_tk_extra
            )
            # sys.stdout = sys.__stdout__
        except Exception as e:
            print(f"  Error loading data for {session}: {e}")
           
    else:
        try:
            # sys.stdout = open(os.devnull, 'w') 
            _, ripples,duration_s = load_experimental_data(
                data_path_original,
                session,
                channel=1,
                load_data=False, 
                verbose=False,
                data_reader=read_data
            )
            # sys.stdout = sys.__stdout__
        except Exception as e:
            print(f"  Error loading data for {session}: {e}")

    result={
    "Session": session,
    "Duration_s": duration_s,
    "Num_Ripples": len(ripples),
    }
    
    
    return result


if __name__ == "__main__":
    # Configuration
    # Path to the directory containing spike pickle files (searches recursively)
    SPIKES_ROOT = os.path.join(ROOT_DIR, "snnTorch", "generalization_madrid", "spikes")
    # DATA_PATH_ORIGINAL = r"C:\PedroFelix\Madrid_tests" # Update this if different
    # DATA_PATH_EXTRA = r"C:\PedroFelix\extra_data\original_data" # Update this if different
    DATA_PATH_ORIGINAL = r"E:\NCN\Madrid_tests" # Update this if different
    DATA_PATH_EXTRA = r"E:\NCN\extra_data\original_data" # Update this if different

    print("Looking for network metrics in SPIKES_ROOT:", SPIKES_ROOT)
    
    
    results=[]
    # Walk through all subdirectories
    metrics_df=pd.read_csv(os.path.join(SPIKES_ROOT, "all_networks_metrics.csv"))
    for index,row in metrics_df.iterrows():
        net_name=row["Network"]
        adapt=row["ADAPT"]
        session=row["Session"]
        if net_name=="updnb4ds_100_7" and adapt==0:
            try:
                result = process_live_results(
                    session,
                    DATA_PATH_ORIGINAL,
                    DATA_PATH_EXTRA,
                    )
                results.append(result)
            except Exception as e:
                print(f"  Error processing file: {e}")
                continue
    df=pd.DataFrame(results)
    OUTPUT_FILE = os.path.join(SPIKES_ROOT, f"all_session_durations.xlsx")
    df.to_excel(OUTPUT_FILE, index=False)

