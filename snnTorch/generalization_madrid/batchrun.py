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
    sessions=lists_sessions.annotated_sessions
    channel_sessions=lists_sessions.channel_sessions
    run_inference(
            prefix,
            data_path,
            sessions,
            channel_sessions=channel_sessions,
            adapt=adapt,
            export_spikes=True,
            seed=None)
    gc.collect()