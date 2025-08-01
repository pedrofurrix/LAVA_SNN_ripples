from process_results import *
import pickle
import os
#
adapt=2
test_ds_b4=True
if test_ds_b4:
    identifier=f"testing_dsb4_adaptable{adapt}" if adapt > 0 else "testing_dsb4"
else:
    identifier=f"30000_1000_100_adaptable{adapt}" if adapt > 0 else "30000_1000_100"
    # identifier="1000_200_median"
parent_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir,os.pardir)
dataset_path=os.path.join(parent_dir,"extract_Nripples","train_pedro","dataset_up_down",identifier)

networks=["dsb4updn_median_200_15f",
          "dsb4updn_median_200_11b",
          "dsb4updn_median_200_13f",
          "dsb4updn_median_200_9f",
          "dsb4updn_median_200_12b",
          "updnb4ds_100_13b",
          "updnb4ds_100_7",
            "updnb4ds_100_13f",
            "adapt20_3b",
            "dsb4updn_median_200_11f",
            "dsb4updn_median_200_14b",
            "dsb4updn_median_200_16b"]

spikes_names= [f"{net}_adapt{adapt}" for net in networks if adapt > 0] if adapt > 0 else networks
spikes_names = [f"{net}_dsb4" for net in networks] if test_ds_b4 else spikes_names

# Example list of networks
list_nets=[
    {
        "name": net,
        "dataset_path": dataset_path,
        "spikes": os.path.join(os.path.dirname(os.path.abspath(__file__)),"spikes",f"{spike}_spikes.npy")
    }
    for net,spike in zip(networks,spikes_names)
]

all_results = {}

for net in list_nets:
    print(f"Processing {net['name']}")
    info = process_network_results(
        net["dataset_path"],
        net["spikes"],
        tolerance=20,
        padding=100,
        refractory_period=0,
        max_detection_offset=80,
    )
    all_results[net["name"]] = info

# Save all results to a pickle file
results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"live_results")
os.makedirs(results_dir, exist_ok=True)
filename= f"all_network_results_{identifier}.pkl"
with open(os.path.join(results_dir, filename), "wb") as f:
    pickle.dump(all_results, f)

print(f"Results saved to {os.path.join(results_dir, filename)}")