from live_validation_batcher import run_inference
import argparse
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SNN model for ripple detection.")
    parser.add_argument("--prefix", type=str, default="dsb4updn_median_200_15f", help="Prefix of the trained model file.")
    parser.add_argument("--identifier", type=str, default=None, help="Identifier for dataset path.")
    parser.add_argument("--adapt", type=int, default=120, help="Adaptation parameter.")
    parser.add_argument("--test_dsb4", action="store_true", help="Flag to indicate test on dsb4 dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    # args = parse_args()

    # Unpack args
    # prefix = args.prefix
    # adapt = args.adapt
    # test_ds_b4 = args.test_dsb4
    ids_networks=[4,5,6]
    ids=[0,1]
    common_identifier = "iiss"
    prefixes=[f"{common_identifier}_{idn}b" for idn in ids_networks] + [f"{common_identifier}_adapt20_1b",f"{common_identifier}_adapt20_2b"]
    adapts=[0,20,60,120]
    for prefix in prefixes:
        for adapt in adapts:
            run_inference(prefix, ids, adapt, export_spikes=True, seed=None,min_threshold=0)
    gc.collect()