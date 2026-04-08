import argparse
import gc
import os
from live_validation_batcher import run_inference
import sys
ROOT_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in os.sys.path:
    sys.path.append(ROOT_DIR)
import liset_data_reader.lists_sessions as lists_sessions



if __name__ == "__main__":
    
    # Unpack args
    prefix = "updnb4ds_100_7"
    adapt = 20

    data_path=r"C:\PedroFelix\extra_data\original_data"
    
    # sessions=lists_sessions.annotated_sessions

#     # Original Sessions 
#     session_set={"2025-09-22_17-55-26", #R
#                  "2025-09-23_15-50-26", #R
#                  "2025-09-24_10-24-40", #R
#                  "2025-09-25_16-41-14"} #R

#     # Extra
# #     session_set={"2025-09-24_16-29-07", #R   
# #             "2025-09-24_17-38-17",} #R 
#     session_set.update({ "2025-09-24_16-29-07", #R   
#             "2025-09-24_17-38-17",
#                 "2025-09-22_17-42-27", 
#                  "2025-09-23_16-17-52", 
#                  "2025-09-24_11-34-51",
#                  "2025-09-25_11-21-53",
#                  "2025-09-25_12-52-22",})
    session_set=    {
            "Calbai32FPGA_251003_144832",
            # "Calbai32FPGA_251003_150055",
            # "PV01ai32FPGA_250611_115923",
            # "PV01ai32FPGA_250611_122326",
            # "Calb_251209_160255",
            # "Calb_251210_115904",
            # "Calb_251210_121141",
            # "Calb_251210_122327",
            # "Calb_251210_162849",
            # "Calb_251210_164150",
            # "Calb_251210_165332",
            # "Calb_251211_104316",
            # "Calb_251211_105518",
            # "Calb_251211_110650",
                } 
    channel_sessions=lists_sessions.channel_sessions
    run_inference(
            prefix,
            data_path,
            session_set,
            channel_sessions=channel_sessions,
            adapt=adapt,
            export_spikes=False,
            seed=None)
    gc.collect()