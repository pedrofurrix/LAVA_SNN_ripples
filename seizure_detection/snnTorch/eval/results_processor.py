from process_results import *
import pickle
import os
#
adapt=[0,20,60,120]
for adapt in adapt:
    parent_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir,os.pardir)
    dataset_path=os.path.join(parent_dir,"up_dn_spikes","live_validation",f"adapt_{adapt}")

    ids_networks=[1,2,3,4,5,6]
    common_identifier = "iiss"
    networks=[f"{common_identifier}_{idn}b" for idn in ids_networks] # + [f"{common_identifier}_{idn}f" for idn in ids_networks]

    spikes_names= [f"{net}_adapt{adapt}" for net in networks if adapt > 0] if adapt > 0 else networks

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
            tolerance=0,
            padding=100,
            refractory_period=0,
            max_detection_offset=100,
        )
        all_results[net["name"]] = info

    # Save all results to a pickle file
    results_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"live_results")
    os.makedirs(results_dir, exist_ok=True)
    filename= f"all_network_results_adapt_{adapt}.pkl"
    with open(os.path.join(results_dir, filename), "wb") as f:
        pickle.dump(all_results, f)

    print(f"Results saved to {os.path.join(results_dir, filename)}")