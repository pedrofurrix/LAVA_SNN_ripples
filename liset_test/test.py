import sys
import os
import pickle as pkl
curr_dir=os.path.dirname(os.path.abspath(__file__))
par_dir=os.path.dirname(curr_dir)
sys.path.insert(0, par_dir)

from liset_tk.load_data import *
from liset_tk.read_data import *
import liset_tk.lists_sessions as lists_sessions
from liset_test.process_signal import *
from liset_tk.format_predictions import *

import tensorflow.keras.backend as K
import tensorflow.keras as kr
import tensorflow as tf 
model = kr.models.load_model(os.path.join(curr_dir, "model"), compile=False)
# model.compile(loss="binary_crossentropy", optimizer=optimizer)
model.summary()
print(tf.config.list_physical_devices('GPU'))