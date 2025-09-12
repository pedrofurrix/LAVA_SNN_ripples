import torch
import numpy as np

def evaluate_model(test_loader, net, forward_step, device,detection_window,verbose_training=False,only_spiked=False):
    test_loss_hist = []         # History of the test loss
    test_acc_hist = []          # History of the test accuracy
       # For debugging purposes
    max_test_iter = None        # Maximum number of iterations to test the network
    first_spike_times = []       # List to store the first spike times for each sample

    TP, FP, FN, TN = 0, 0, 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass, now returns spike trains
            loss_val, acc_val, out_spikes = forward_step(data, targets, is_train=False, return_spikes=True,verbose=False)

            test_loss_hist.append(loss_val)
            test_acc_hist.append(acc_val)

            batch_size = data.shape[0]
            num_steps = out_spikes.shape[0]

            # Assume GT format: each item in targets is the index of the spike time, or -1 if no spike
            for b in range(batch_size):
                pred_spikes = out_spikes[:, b]  # Single output neuron
                first_spike = torch.argmax(pred_spikes, dim=0) # First spike times for each neuron in the batch
                # If max spike value is 0 -> set the first spike to -1   
                first_spike=-1 if first_spike == 0 else first_spike.item()
                first_spike_times.append(first_spike)
                gt_time = targets[b].item()
                if gt_time == -1:
                    # No event in ground truth, check if any spikes fired
                    if first_spike !=-1:
                        FP += 1  # Spurious detection
                    else:
                        TN += 1  # Correct no-event prediction
                else:
                    # There is a GT event
                    # Check if any spike occurred in the detection window
                    if only_spiked and first_spike != -1:
                        detected=True
                    else:
                        detected = first_spike != -1 and (gt_time-detection_window[0] <= first_spike <= gt_time + detection_window[1])
                    # detected=first_spike!=-1
                    if detected:
                        TP += 1
                    else:
                        FN += 1
    return TP, FP, FN, TN, test_loss_hist, test_acc_hist,torch.tensor(first_spike_times)

def permutation_test(test_loader,first_spike_times,detection_window, N=1000):
   
    
    all_targets = []
    for _, targets in test_loader:
        all_targets.append(targets)
    all_targets = torch.cat(all_targets, dim=0)


    # Compute actual score (same as before)
    TP, FP, FN = 0, 0, 0
    for i, gt_time in enumerate(all_targets):
        first_spike = first_spike_times[i].item()

        if gt_time == -1:
            if first_spike != -1:
                FP += 1
        else:
            detected = first_spike != -1 and (gt_time - detection_window[0] <= first_spike <= gt_time + detection_window[1])
            if detected:
                TP += 1
            else:
                FN += 1
    actual_score = 2 * TP / (2 * TP + FP + FN)

    null_distribution = []
    for _ in range(N):
        permuted_targets = all_targets[torch.randperm(len(all_targets))]
        TP, FP, FN = 0, 0, 0
        for i, gt_time in enumerate(permuted_targets):
            first_spike = first_spike_times[i].item()
            if gt_time == -1:
                if first_spike != -1:
                    FP += 1
            else:
                detected = first_spike != -1 and (gt_time - detection_window[0] <= first_spike <= gt_time + detection_window[1])
                if detected:
                    TP += 1
                else:
                    FN += 1
        perm_score = 2 * TP / (2 * TP + FP + FN)
        null_distribution.append(perm_score)

    p_value = np.mean([s >= actual_score for s in null_distribution])
    return actual_score, null_distribution, p_value



def permutation_test_accurate_targets(test_loader,first_spike_times,detection_window, N=1000):
   
    all_targets = []
    for _, targets in test_loader:
        all_targets.append(targets)
    all_targets = torch.cat(all_targets, dim=0)


    # Compute actual score (same as before)
    TP, FP, FN = 0, 0, 0
    for i, gt_time in enumerate(all_targets):
        first_spike = first_spike_times[i].item()
        
        if gt_time == -1:
            if first_spike != -1:
                FP += 1
        else:
            detected = first_spike != -1 and (gt_time - detection_window[0] <= first_spike <= gt_time + detection_window[1])
            if detected:
                TP += 1
            else:
                FN += 1
    actual_score = 2 * TP / (2 * TP + FP + FN)

    null_distribution = []
    for _ in range(N):
        permuted_targets = all_targets[torch.randperm(len(all_targets))]
        TP, FP, FN = 0, 0, 0
        for i, gt_time in enumerate(permuted_targets):
            first_spike = all_targets[i].item()
            if gt_time == -1:
                if first_spike != -1:
                    FP += 1
            else:
                detected = first_spike != -1 and (gt_time - detection_window[0] <= first_spike <= gt_time + detection_window[1])
                if detected:
                    TP += 1
                else:
                    FN += 1
        perm_score = 2 * TP / (2 * TP + FP + FN)
        null_distribution.append(perm_score)

    p_value = np.mean([s >= actual_score for s in null_distribution])
    return actual_score, null_distribution, p_value

def permutation_test_accurate_targets_no_pred(test_loader,detection_window, N=1000):
   
    all_targets = []
    for _, targets in test_loader:
        all_targets.append(targets)
    all_targets = torch.cat(all_targets, dim=0)


    # # Compute actual score (same as before)
    # TP, FP, FN = 0, 0, 0
    # for i, gt_time in enumerate(all_targets):
    #     first_spike = first_spike_times[i].item()
        
    #     if gt_time == -1:
    #         if first_spike != -1:
    #             FP += 1
    #     else:
    #         detected = first_spike != -1 and (gt_time - detection_window[0] <= first_spike <= gt_time + detection_window[1])
    #         if detected:
    #             TP += 1
    #         else:
    #             FN += 1
    # actual_score = 2 * TP / (2 * TP + FP + FN)

    null_distribution = []
    for _ in range(N):
        permuted_targets = all_targets[torch.randperm(len(all_targets))]
        TP, FP, FN = 0, 0, 0
        for i, gt_time in enumerate(permuted_targets):
            first_spike = all_targets[i].item()
            if gt_time == -1:
                if first_spike != -1:
                    FP += 1
            else:
                detected = first_spike != -1 and (gt_time - detection_window[0] <= first_spike <= gt_time + detection_window[1])
                if detected:
                    TP += 1
                else:
                    FN += 1
        perm_score = 2 * TP / (2 * TP + FP + FN)
        null_distribution.append(perm_score)

    # p_value = np.mean([s >= actual_score for s in null_distribution])
    return null_distribution


import random


def generate_shuffled_events(ripple_starts, T):
    inter_event_intervals = ripple_starts[1:] - ripple_starts[:-1]
    shuffled_iei = inter_event_intervals[torch.randperm(len(inter_event_intervals))]

    # Random start, ensuring total fits in T
    start = random.randint(0, T - shuffled_iei.sum().item() - 1)
    shuffled_event_times = [start]
    for dt in shuffled_iei:
        shuffled_event_times.append(shuffled_event_times[-1] + dt.item())
    return torch.tensor(shuffled_event_times)

def apply_refractory(output_spikes, refrac_period):
    """Keep only the first spike, then suppress any within the refractory window."""
    output_spikes = sorted(output_spikes)
    filtered_spikes = []
    last_spike_time = -float('inf')

    for t in output_spikes:
        if t - last_spike_time >= refrac_period:
            filtered_spikes.append(t)
            last_spike_time = t
    return filtered_spikes


def evaluate_live_performance(output_spikes, ripple_times, detection_window,refrac_period=200):
    output_spikes = apply_refractory(output_spikes, refrac_period)
    output_spikes = torch.tensor(output_spikes)
  
    TP, FP, FN = 0, 0, 0
    used_spikes = set()

    for gt_time in ripple_times:
        detected = False
        for i, spike_time in enumerate(output_spikes):
            if i in used_spikes:
                continue
            if gt_time <= spike_time <= gt_time + detection_window:
                TP += 1
                used_spikes.add(i)
                detected = True
                break
        if not detected:
            FN += 1

    FP = len(output_spikes) - len(used_spikes)
    return TP, FP, FN

def live_permutation_test(ripple_starts,window_size, output_spikes,detection_window,refrac_period=200, N=1000):
    ## Evaluate actual performance
    output_spikes = torch.tensor(output_spikes)
    TP,FP,FN=evaluate_live_performance(output_spikes, ripple_starts, detection_window,refrac_period)
    actual_score = 2 * TP / (2 * TP + FP + FN)

    null_distribution = []
    total_timesteps = window_size
    shuffled_event_times= generate_shuffled_events(ripple_starts, window_size)
    for _ in range(N):
        # Shuffle the event times
        shuffled_event_times = generate_shuffled_events(ripple_starts, total_timesteps)
        TP, FP, FN = 0, 0, 0

        # Evaluate performance on shuffled events
        TP, FP, FN = evaluate_live_performance(output_spikes, shuffled_event_times, detection_window,refrac_period)

        perm_score = 2 * TP / (2 * TP + FP + FN)
        null_distribution.append(perm_score)
    p_value = np.mean([s >= actual_score for s in null_distribution])
    return actual_score, null_distribution, p_value