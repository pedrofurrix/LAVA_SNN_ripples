import torch

def plot_sample_spikes(model, test_loader, num_steps, sample_idx=None):
    """Interactive spike visualization with label and prediction display.
    
    Args:
        model: Trained SNN
        test_loader: DataLoader with test data
        num_steps: Number of timesteps
        sample_idx: Optional specific sample index. If None, chooses random sample.
    """
    # Get the test dataset from loader
    test_data = test_loader.dataset
    
    # Select sample
    if sample_idx is None:
        sample_idx = torch.randint(0, len(test_data), (1,)).item()
    
    sample_data, true_label = test_data[sample_idx]
    sample_data = sample_data.to(device).unsqueeze(0)  # add batch dim
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        spk_rec, mem_rec = forward_pass(model, num_steps, sample_data)
    
    # Get prediction (sum spikes over time)
    predicted_label = spk_rec.sum(dim=0).argmax().item()
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot membrane potential
    plt.subplot(3, 1, 1)
    plt.plot(mem_rec[:, 0, :].cpu().T)  # all output neurons
    plt.title(f"Sample {sample_idx} - True: {'Positive' if true_label else 'Negative'} | "
              f"Predicted: {'Positive' if predicted_label else 'Negative'}")
    plt.ylabel("Membrane Potential")
    plt.legend([f"Neuron {i}" for i in range(mem_rec.shape[2])])
    
    # Plot output spikes
    plt.subplot(3, 1, 2)
    for i in range(spk_rec.shape[2]):  # for each output neuron
        spikes = spk_rec[:, 0, i].cpu().nonzero()[:, 0].numpy()
        if len(spikes) > 0:
            plt.eventplot(spikes, lineoffsets=i+1, linewidths=0.5)
    plt.yticks([1, 2], ["Negative Neuron", "Positive Neuron"])
    plt.ylabel("Output Spikes")
    
    # Plot input spikes (assuming 2 input channels)
    plt.subplot(3, 1, 3)
    for i in range(2):  # up/down channels
        input_spikes = sample_data[0, i].cpu().numpy()
        plt.eventplot(np.where(input_spikes > 0)[0], lineoffsets=i+1, linewidths=0.5)
    plt.yticks([1, 2], ["Input Down", "Input Up"])
    plt.xlabel("Time Step")
    plt.ylabel("Input Spikes")
    
    plt.tight_layout()
    plt.show()
    
    return sample_idx

# Interactive usage:
def explore_samples(model, test_loader, num_steps):
    """Interactive sample explorer."""
    current_idx = 0
    total_samples = len(test_loader.dataset)
    
    while True:
        print(f"\nSample {current_idx}/{total_samples-1}")
        current_idx = plot_sample_spikes(model, test_loader, num_steps, current_idx)
        
        user_input = input("Next [n], Previous [p], Jump to index [0-9], Quit [q]: ")
        if user_input.lower() == 'n':
            current_idx = (current_idx + 1) % total_samples
        elif user_input.lower() == 'p':
            current_idx = (current_idx - 1) % total_samples
        elif user_input.isdigit():
            new_idx = int(user_input)
            if 0 <= new_idx < total_samples:
                current_idx = new_idx
        elif user_input.lower() == 'q':
            break

# Usage:
explore_samples(net, test_loader, num_steps)