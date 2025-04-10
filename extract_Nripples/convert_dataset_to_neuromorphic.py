import matplotlib.pyplot as plt
import numpy as np
import sys
import os
liset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../liset_tk'))
sys.path.insert(0, liset_path)
from signal_aid import y_discretize_1Dsignal, cutoff_amplitude
import os

bandpass=[100,250]
# Load the saved dataset
data_dir = 'train_pedro/dataset'
true_positives = np.load(os.path.join(data_dir, f'true_positives_{bandpass[0]}_{bandpass[1]}Hz.npy'))
true_negatives = np.load(os.path.join(data_dir, f'true_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy'))
# y_size = int(sys.argv[1])
y_size=20

# Define parameters
samples_len = true_positives.shape[1]
y_num_samples = y_size
cutoff = cutoff_amplitude(true_positives)

num_samples = true_positives.shape[0] #1794 # length of the true positives # acho que devia ser true_positives.shape[0]

n_true_positives = np.zeros((num_samples, y_num_samples, samples_len))
n_true_negatives = np.zeros((num_samples, y_num_samples, samples_len))

for idx in range(true_positives.shape[0]):
    n_true_positives[idx, :, :] = y_discretize_1Dsignal(true_positives[idx], y_num_samples, cutoff)
    n_true_negatives[idx, :, :] = y_discretize_1Dsignal(true_negatives[idx], y_num_samples, cutoff)  

# Save the arrays
save_dir=data_dir = os.path.join("train_pedro","n_dataset","marcos",f"{y_size}")
os.makedirs(save_dir, exist_ok = True)
np.save(os.path.join(save_dir,f"ntrue_positives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=n_true_positives, allow_pickle=True)
np.save(os.path.join(save_dir,f"ntrue_negatives_{bandpass[0]}_{bandpass[1]}Hz.npy"), arr=n_true_negatives, allow_pickle=True)