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


from snnTorch.retry_live_val.process_signal import load_experimental_data

from liset_data_reader.liset_paper import liset_paper


def process_live_results(
    data_path,
    session,
):
    path_data=os.path.join(data_path, session)
    channel=1

    try:
        # Suppress stdout from load_experimental_data if verbose
        # sys.stdout = open(os.devnull, 'w') 
            _, ripples,duration_s = load_experimental_data(
            path=path_data,
            channel=channel,
            load_data=False, 
            verbose=True,
            downsample=False,
            fraction=(0,1),
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
    DATA_PATH=r"E:\NCN\neurospark_mat\Download_from_paper"
    sessions=os.listdir(DATA_PATH)
    RESULT_DIR=os.path.join(os.path.dirname(__file__), "spikes")
    os.makedirs(RESULT_DIR, exist_ok=True)

    results=[]
    # Walk through all subdirectories
    for session in sessions: 
        try:
            result = process_live_results(
                DATA_PATH,
                session
                )
            results.append(result)
        except Exception as e:
            print(f"  Error processing file: {e}")
            continue
    df=pd.DataFrame(results)

    OUTPUT_FILE = os.path.join(RESULT_DIR, f"all_session_durations.xlsx")
    df.to_excel(OUTPUT_FILE, index=False)

