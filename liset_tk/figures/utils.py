import numpy as np
import matplotlib.pyplot as plt

def get_dataset_info(liset):
    ripples= liset.ripples_GT
    numsamples=liset.data.shape[0]
    print("Dataset Time:",round(numsamples / liset.fs,2), "seconds")
    time_seconds = numsamples / liset.fs
    minutes= np.floor(time_seconds/ 60)
    extra_seconds = time_seconds%60
    print("Dataset time:", int(minutes), "minutes", round(extra_seconds,2), "seconds")
    
    # Ripple Analysis
    rate = ripples.shape[0] / time_seconds  # Ripples per second
    ripples = ripples[np.argsort(ripples[:, 0])]
    total_ripple_samples= np.sum(ripples[:, 1] - ripples[:, 0])   # Total ripple time in seconds
   
    print("Number of ripples:", ripples.shape[0])
    print("Ripples per minute:", round(ripples.shape[0] / time_seconds*60, 2))
    print("Ripples per second:", round(rate, 2))
    print("Seconds per ripple:", round(time_seconds / ripples.shape[0], 2))
    print("Total Ripple Time:", round(total_ripple_samples/ liset.fs, 2), "seconds")
    print("Percentage of time with ripples:", round(total_ripple_samples / numsamples * 100, 2), "%")


    durations = (ripples[:, 1] - ripples[:, 0])/liset.fs*1000 # Convert to milliseconds
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    print(f"Mean Duration: {mean_duration:.2f} ms, Std Duration: {std_duration:.2f} ms, Rate: {rate:.2f} ripples/s")
    distance_between=[ripples[i, 0] - ripples[i-1, 1] for i in range(1, ripples.shape[0])]

def pie_ripples(liset):
    # Pie Chart for Ripples
    ripples= liset.ripples_GT
    total_samples= liset.data.shape[0]
    total_ripple_samples= np.sum(ripples[:, 1] - ripples[:, 0])
    total_non_ripple_samples= total_samples - total_ripple_samples
    labels = ['Ripples', 'Non-Ripples']
    sizes = [total_ripple_samples, total_non_ripple_samples]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  # explode the first slice
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
            shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Ripple vs Non-Ripple Samples")
    plt.show()

def plot_ripple_durations(liset):
    ripples = liset.ripples_GT
    durations = (ripples[:, 1] - ripples[:, 0]) / liset.fs * 1000  # Convert to milliseconds
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    median_duration = np.median(durations)
    percentile75_duration = np.percentile(durations, 75)
    percentile25_duration = np.percentile(durations, 25)
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, color='blue', alpha=0.7)
    plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_duration:.2f} ms')
    plt.axvline(median_duration, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_duration:.2f} ms')
    plt.axvline(percentile75_duration, color='black', linestyle='-', linewidth=0.5, label=f'75th Percentile: {percentile75_duration:.2f} ms')
    plt.axvline(percentile25_duration, color='black', linestyle='-', linewidth=0.5, label=f'25th Percentile: {percentile25_duration:.2f} ms')
    plt.title('Ripple Durations')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

