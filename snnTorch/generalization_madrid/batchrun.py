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
    args = parse_args()

    # Unpack args
    prefix = args.prefix
    adapt = args.adapt
    test_ds_b4 = args.test_dsb4
    if args.identifier:
        identifier = args.identifier
        if adapt > 0:
            identifier += f"_adaptable{adapt}"
    else:
        if args.test_dsb4:
            identifier = f"testing_dsb4_adaptable{args.adapt}" if args.adapt > 0 else "testing_dsb4"
        else:
            identifier = f"30000_1000_100_adaptable{args.adapt}" if args.adapt > 0 else "30000_1000_100"

    run_inference(prefix, identifier, adapt=adapt, test_dsb4=test_ds_b4, export_spikes=True, seed=None)
    gc.collect()