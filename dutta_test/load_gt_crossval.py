import os
import re
import pickle 
import numpy as np
class CrossValLoader:
    def __init__(self, data_dir, model,threshold,tolerance=100.0):
        self.data_dir = data_dir
        self.model = model
        self.threshold=threshold
        self.tolerance = tolerance
        self.load_ground_truth() 
        self.refine_GTs(tolerance_ms=self.tolerance)     
        # Additional initialization code here

    def load_ground_truth(self):
        # Code to load ground truth data for cross-validation
        pass

    def refine_GTs(self, tolerance_ms=100.0):
        """
        Merges spikes/events that are closer than tolerance_ms into a single event 
        (taking the first timestamp of the burst).
        """
        if not hasattr(self, 'detections'):
            return
        self.GT={}
        for session, spikes in self.detections.items():
            if len(spikes) == 0:
                self.detections[session] = np.array([])
                continue
            
            spikes = np.sort(spikes)
            merged_spikes = []
            
            # Initialize with the first spike of the first group
            group_start = spikes[0]
            prev_spike = spikes[0]
            
            for spike in spikes[1:]:
                # If the gap is larger than tolerance, close the previous group and start a new one
                if spike - prev_spike >= tolerance_ms:
                    merged_spikes.append(group_start)
                    group_start = spike
                
                # Update prev_spike to check chaining for the next iteration
                prev_spike = spike
            
            # Append the last group start
            merged_spikes.append(group_start)
            
            self.GT[session] = np.array(merged_spikes)
class RippleNet_GT(CrossValLoader):
    def __init__(self, data_dir= r"C:\Users\NCN\Documents\PedroFelix\RippleNet\test_madrid\predictions", model= "ripplenet_bidirectional_best_random_seed456", threshold=0.8,tolerance=100.0):
        super().__init__(data_dir, model=model, threshold=threshold, tolerance=tolerance)
        # Additional initialization code specific to RippleNET can go here
    def load_ground_truth(self):
        ground_truth= {}

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".pkl") and "_predictions_" in file:
                    # Parse session and threshold
                    # Format: {session}_predictions_{threshold}.pkl
                    # Regex
                    match = re.match(r"(.+)_predictions_([\d\.]+)\.pkl", file)
                    if match:
                        session = match.group(1)
                        threshold = float(match.group(2))
                        
                        pkl_path = os.path.join(root, file)

                        if threshold == self.threshold:
                            with open(pkl_path, 'rb') as f:
                                session_predictions = pickle.load(f)    
                            for model_name, data in session_predictions.items():
                                if model_name == self.model:
                                    print(f"Loading detections for session {session}, model {model_name}, threshold {threshold}")
                                    predictions_s = data['predictions_time']
                                    spikes_ms = predictions_s * 1000.0 # ms
                                    ground_truth[session] = spikes_ms
                                    break
                                else:
                                    continue
        self.detections=ground_truth


class RipplAI_GT(CrossValLoader):
    def __init__(self, data_dir= r"C:\Users\NCN\Documents\PedroFelix\rippl-AI\detections", model= "LSTM_5", threshold=0.3,tolerance=100.0):
        super().__init__(data_dir, model=model, threshold=threshold, tolerance=tolerance)
        # Additional initialization code specific to RippleNET can go here
    
    def load_ground_truth(self):
        ground_truth= {}
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".pkl") and "SWR_detections_" in file:

                    match = re.match(r"SWR_detections_(.+)_th([\d\.]+)\.pkl", file)
                    if match:
                        session = match.group(1)
                        threshold = float(match.group(2))
                        
                        pkl_path = os.path.join(root, file)
                        if threshold == self.threshold:
                            with open(pkl_path, 'rb') as f:
                                session_predictions = pickle.load(f)

                            for model_name, detections in session_predictions.items():
                                if model_name == self.model:
                                    print(f"Loading detections for session {session}, model {model_name}, threshold {threshold}")
                                    if len(detections) == 0:
                                        spikes_ms = np.array([])
                                    else:
                                        # Use start of interval
                                        # middle_samples = np.mean(detections, axis=1)
                                        # spikes_ms = (middle_samples / 1250.0) * 1000.0 # ms
                                        start_samples = detections[:, 0]
                                        spikes_ms = (start_samples / 1250) * 1000.0 # ms
                                        ground_truth[session] = spikes_ms
                                        break
                                else:
                                    continue
        self.detections=ground_truth

class LisetCNN_GT(CrossValLoader):
    def __init__(self, data_dir= r"C:\Users\NCN\Documents\PedroFelix\LAVA_SNN_ripples\liset_test\cnn_detections", model= None, threshold=0.7,tolerance=100.0):
        super().__init__(data_dir, model=model, threshold=threshold, tolerance=tolerance)
        # Additional initialization code specific to RippleNET can go here
    
    def load_ground_truth(self):
        ground_truth= {}
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".pkl") and "detections" in file:
                    pkl_path = os.path.join(root, file)
                    # print(f"\nFound results file: {file}")
                    
                    # Determine network name from filename or folder
                    # e.g. detections_thr_0.6.pkl
                    net_name = file.replace('.pkl', '')
                    
                    # Use regex to find thr_X.X suffix
                    match = re.search(r'thr_([\d\.]+)', net_name)
                    threshold = float(match.group(1)) if match else 0
                    if threshold == self.threshold:
                        print(f"Loading detections for threshold {threshold}")
                        with open(pkl_path, 'rb') as f:
                            predictions = pickle.load(f)
                        for session, data in predictions.items():
                            if 'pred_times' in data:
                                spikes_ms = data['pred_times'] * 1000 # Convert s to ms
                            else:
                                print(f"Warning: No pred_times for session {session}")
                                spikes_ms = np.array([])
                            spikes_ms=spikes_ms[:,0] if len(spikes_ms.shape)>1 else spikes_ms
                            ground_truth[session] = spikes_ms
        self.detections=ground_truth

class Dutta_GT(CrossValLoader):
    def __init__(self, data_dir= r"C:\Users\NCN\Documents\PedroFelix\LAVA_SNN_ripples\dutta_test", model= None, threshold=5.0,tolerance=100.0):
        super().__init__(data_dir, model=model, threshold=threshold, tolerance=tolerance)
        # Additional initialization code specific to RippleNET can go here
    
    def load_ground_truth(self):
        ground_truth= {}
         # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".pkl") and "detections" in file:
                    pkl_path = os.path.join(root, file)
                    # print(f"\nFound results file: {file}")
                    
                    # Determine network name from filename or folder
                    # e.g. detections_thr_3.0.pkl
                    net_name = file.replace('.pkl', '')
                    
                    # Use regex to find thr_X.X suffix
                    match = re.search(r'thr_([\d\.]+)', net_name)
                    threshold = float(match.group(1)) if match else 0
                    if threshold == self.threshold:
                            print(f"Loading detections for threshold {threshold}")
                            with open(pkl_path, 'rb') as f:
                                predictions = pickle.load(f)
                            for session, data in predictions.items():
                                detections = data.get('detections', [])
                                spikes_ms = np.array([d['time_s'] * 1000 for d in detections]) # Convert s to ms
                                ground_truth[session] = spikes_ms
        self.detections=ground_truth


class SNN_GT(CrossValLoader):
    def __init__(self, data_dir= r"C:\Users\NCN\Documents\PedroFelix\LAVA_SNN_ripples\snnTorch\generalization_madrid\spikes", model= "updnb4ds_100_13f", threshold=20,tolerance=100.0):
        super().__init__(data_dir, model=model, threshold=threshold,tolerance=tolerance)
        # Additional initialization code specific to RippleNET can go here
    
    def load_ground_truth(self):
        ground_truth= {}
         # Walk through all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith("_spikes.pkl"):
                    pkl_path = os.path.join(root, file)
                    # print(f"\nFound results file: {file}")
                    
                    # Determine network name from filename or folder
                    # e.g. updnb4ds_100_7_adapt0_spikes.pkl -> updnb4ds_100_7_adapt0
                    net_name = file.replace('_spikes.pkl', '')
                    
                    # Use regex to find _adaptX suffix
                    # This avoids matching "adapt" inside the network name (e.g. "adapt20")
                    match = re.search(r'_adapt(\d+)$', net_name)
                    adapt = int(match.group(1)) if match else 0
                    # Remove the _adaptX suffix from the network name
                    net_name = re.sub(r'_adapt\d+$', '', net_name) if match else net_name
                    if net_name == self.model and adapt == self.threshold:
                        print(f"Loading spikes for model {net_name} with adapt {adapt}")
                        with open(pkl_path, 'rb') as f:
                            predictions = pickle.load(f)
                        for session, data in predictions.items():
                            spikes_ms = data.get('spikes', np.array([]))
                            ground_truth[session] = spikes_ms
                        break
        self.detections=ground_truth