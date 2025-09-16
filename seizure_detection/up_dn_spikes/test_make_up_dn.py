from make_up_dn_iis import make_up_dn_dataset
# parent=r"E:\neurospark_mat\KA MODEL TRANSITION SESSIONS"
parent=r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\KA MODEL TRANSITION SESSIONS"
ids=[2]

downsampled_fs=1000
bandpass=[1,70]
window_size=0.2
sample_ratio=0.25
scaling_factor=1.5
percentile=False
refractory=False

adapt_threshold=True

time_max=[120,60,20]
for time in time_max:
    overlap=int(0.5*time)
    # window=[1500,0]
    window=[0,1700]
    make_up_dn_dataset(parent, ids, time, downsampled_fs, bandpass, window_size, sample_ratio, scaling_factor, percentile, refractory, overlap,adapt_threshold=adapt_threshold,window=window)