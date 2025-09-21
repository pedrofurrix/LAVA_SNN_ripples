import os
import pickle
import numpy as np
import pandas as pd

def compute_metrics(info):
    """
    Compute performance metrics from process_network_results output.
    Expects dict with TP, FP, FN keys.
    """
    TP = info.get("TP", 0)
    FP = info.get("FP", 0)
    FN = info.get("FN", 0)
    TN = info.get("TN", 0)  # optional, usually 0

    total = TP + FP + FN + TN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / total if total > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }

adapt_levels = [0, 20, 60, 120]

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir,"live_results2")
metrics_dir = os.path.join(results_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

summary_all = []

for adapt in adapt_levels:
    filename = f"all_network_results_adapt_{adapt}.pkl"
    path = os.path.join(results_dir, filename)

    with open(path, "rb") as f:
        all_results = pickle.load(f)

    for net, datasets in all_results.items():
        for dataset, shanks in datasets.items():
            # always shank_2 in your case
            shank_info = shanks.get("shank_1", {})
            metrics = compute_metrics(shank_info)

            row = {"adapt": adapt, "network": net, "dataset": dataset}
            row.update(metrics)
            summary_all.append(row)

# --- Save summary ---
df = pd.DataFrame(summary_all)
csv_file = os.path.join(metrics_dir, "metrics_summary.csv")
df.to_csv(csv_file, index=False)

with open(os.path.join(metrics_dir, "metrics_summary.pkl"), "wb") as f:
    pickle.dump(df, f)

print(f"📊 Summary table saved to {csv_file}")
