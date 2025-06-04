import torch
import numpy as np

def evaluate_model(test_loader, net, forward_step, device,detection_window,verbose_training=False):
    test_loss_hist = []         # History of the test loss
    test_acc_hist = []          # History of the test accuracy
       # For debugging purposes
    max_test_iter = None        # Maximum number of iterations to test the network

    net.eval()
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
                    detected = first_spike != -1 and (gt_time-detection_window <= first_spike <= gt_time + detection_window)
                    # detected=first_spike!=-1
                    if detected:
                        TP += 1
                    else:
                        FN += 1
    return TP, FP, FN, TN, test_loss_hist, test_acc_hist

def permutation_test(test_loader, net, forward_step, device,detection_window, N=1000):
    actual_TP, actual_FP, actual_FN, actual_TN,test_loss,test_acc = evaluate_model(test_loader, net, forward_step, device,detection_window)
    actual_score = 2 * actual_TP / (2 * actual_TP + actual_FP + actual_FN)

    null_distribution = []
    all_targets = []
    for _, targets in test_loader:
        all_targets.append(targets)
    all_targets = torch.cat(all_targets, dim=0)

    with torch.no_grad():
        for _ in range(N):
            permuted_targets_full = all_targets[torch.randperm(all_targets.size(0))]
            TP, FP, FN, TN = 0, 0, 0, 0
            idx=0

            for data, _ in test_loader:
                data = data.to(device)
                batch_size = data.shape[0]
                permuted_targets = permuted_targets_full[idx:idx+batch_size].to(device)
                idx += batch_size

                _,_, out_spikes = forward_step(data, permuted_targets, is_train=False, return_spikes=True)

                # Re-run your confusion matrix calculation here using permuted_targets
                # ...
            batch_size = data.shape[0]
            num_steps = out_spikes.shape[0]

            # Assume GT format: each item in targets is the index of the spike time, or -1 if no spike
            for b in range(batch_size):
                pred_spikes = out_spikes[:, b]  # Single output neuron
                first_spike = torch.argmax(pred_spikes, dim=0) # First spike times for each neuron in the batch
                # If max spike value is 0 -> set the first spike to -1   
                first_spike=-1 if first_spike == 0 else first_spike.item()
                gt_time = permuted_targets[b].item()
                
                if gt_time == -1:
                    # No event in ground truth, check if any spikes fired
                    if first_spike !=-1:
                        FP += 1  # Spurious detection
                    else:
                        TN += 1  # Correct no-event prediction
                else:
                    # There is a GT event
                    # Check if any spike occurred in the detection window
                    detected = first_spike != -1 and (gt_time-detection_window <= first_spike <= gt_time + detection_window)
                    # detected=first_spike!=-1
                    if detected:
                        TP += 1
                    else:
                        FN += 1 
            perm_score = 2 * TP / (2 * TP + FP + FN)
            null_distribution.append(perm_score)

        p_value = np.mean([s >= actual_score for s in null_distribution])
    return actual_score, null_distribution, p_value
