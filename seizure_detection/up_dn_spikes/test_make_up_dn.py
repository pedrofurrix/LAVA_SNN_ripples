from make_up_dn_iis import make_up_dn_dataset
parent=r"E:\neurospark_mat\KA MODEL TRANSITION SESSIONS"
ids=[2,5]
time_max=120
downsampled_fs=1000
bandpass=[1,70]
window_size=0.2
sample_ratio=0.25
scaling_factor=1.5
percentile=False
refractory=False
overlap=int(0.5*time_max)
adapt_threshold=False

make_up_dn_dataset(parent, ids, time_max, downsampled_fs, bandpass, window_size, sample_ratio, scaling_factor, percentile, refractory, overlap,adapt_threshold=adapt_threshold)