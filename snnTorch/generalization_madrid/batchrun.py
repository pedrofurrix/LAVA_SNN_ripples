from live_validation_batcher import run_inference
import argparse
import gc
import lists_sessions

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SNN model for ripple detection.")
    parser.add_argument("--prefix", type=str, default="dsb4updn_median_200_15f", help="Prefix of the trained model file.")
    parser.add_argument("--adapt", type=int, default=120, help="Adaptation parameter.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Unpack args
    prefix = args.prefix
    adapt = args.adapt

    data_path=r"C:\Madrid_tests"
    # sessions=lists_sessions.annotated_sessions

    # Original Sessions 
    # session_set={"2025-09-22_17-55-26", #R
    #              "2025-09-23_15-50-26", #R
    #              "2025-09-24_10-24-40", #R
    #              "2025-09-24_14-22-55", #H
    #              "2025-09-24_15-13-10", #H
    #              "2025-09-25_16-41-14"} #R

    # Extra
#     session_set={"2025-09-24_16-29-07", #R   
#             "2025-09-24_17-38-17",} #R 
    session_set={ "2025-09-22_17-42-27", 
                 "2025-09-23_16-17-52", 
                 "2025-09-24_11-34-51",
                 "2025-09-25_11-21-53",
                 "2025-09-25_12-52-22",}

    channel_sessions=lists_sessions.channel_sessions
    run_inference(
            prefix,
            data_path,
            session_set,
            channel_sessions=channel_sessions,
            adapt=adapt,
            export_spikes=True,
            seed=None)
    gc.collect()