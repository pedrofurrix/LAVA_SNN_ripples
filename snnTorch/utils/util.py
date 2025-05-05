'''
----------- Duration of HFO events -----------
Important Constants regarding HFO events in the Synthetic Dataset
Taken from:https://github.com/monkin77/snn-torch/blob/master/src/snnt_utils/hfo.py 
'''
# HFO Detection Offsets [MIN_OFFSET, MAX_OFFSET, MEAN_OFFSET, TOLERANCE_OFFSET]
RIPPLE_DETECTION_OFFSET = [18, 45, 31, 20] # it's calculated as 4.5 periods of the ripple wavelet - for 100 Hz and 250 Hz as the limit frequencies
# FR_DETECTION_OFFSET = [9, 18, 13, 5]
# BOTH_DETECTION_OFFSET = [9, 57, 33, 24]


# HFO MAX Durations (Not the same as the Offset for SNN Detection)
RIPPLE_MAX_DUR = 120
FR_MAX_DUR = 40

# The Windows for HFO detection are based on the MAX DETECTION OFFSET
RIPPLE_CONFIDENCE_WINDOW = int(round(RIPPLE_DETECTION_OFFSET[1] * 1.8)) 
# FR_CONFIDENCE_WINDOW = int(round(FR_DETECTION_OFFSET[1] * 2.2))
# BOTH_CONFIDENCE_WINDOW = int(round(BOTH_DETECTION_OFFSET[1] * 2.2))