import os
import pickle as pkl
import sys

def join_spikes(spikes_dir, spikes_og_dir):
    # Walk through the spikes_og directory
    for root, dirs, files in os.walk(spikes_og_dir):
        for file in files:
            if file.endswith(".pkl"):
                # Construct the full path for the original file
                og_file_path = os.path.join(root, file)
                
                # Construct the relative path to find the corresponding file in the spikes directory
                rel_path = os.path.relpath(og_file_path, spikes_og_dir)
                target_file_path = os.path.join(spikes_dir, rel_path)
                
                if os.path.exists(target_file_path):
                    print(f"Processing {rel_path}...")
                    
                    # Load the original data (spikes_og)
                    try:
                        with open(og_file_path, "rb") as f:
                            data_og = pkl.load(f)
                    except Exception as e:
                        print(f"Error loading {og_file_path}: {e}")
                        continue
                        
                    # Load the target data (spikes)
                    try:
                        with open(target_file_path, "rb") as f:
                            data_target = pkl.load(f)
                    except Exception as e:
                        print(f"Error loading {target_file_path}: {e}")
                        continue
                    
                    # Find missing sessions
                    added_count = 0
                    for session, session_data in data_og.items():
                        if session not in data_target:
                            data_target[session] = session_data
                            added_count += 1
                            print(f"  Adding session: {session}")
                    
                    if added_count > 0:
                        # Save the updated data back to the target file
                        try:
                            with open(target_file_path, "wb") as f:
                                pkl.dump(data_target, f, protocol=pkl.HIGHEST_PROTOCOL)
                            print(f"  Saved {added_count} new sessions to {target_file_path}")
                        except Exception as e:
                            print(f"Error saving to {target_file_path}: {e}")
                    else:
                        print("  No new sessions found.")
                else:
                    print(f"Target file not found for {rel_path}. Skipping.")

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    spikes_dir = os.path.join(curr_dir, "spikes")
    spikes_og_dir = os.path.join(curr_dir, "spikes_og")
    
    if not os.path.exists(spikes_dir):
        print(f"Directory not found: {spikes_dir}")
        sys.exit(1)
        
    if not os.path.exists(spikes_og_dir):
        print(f"Directory not found: {spikes_og_dir}")
        sys.exit(1)
        
    join_spikes(spikes_dir, spikes_og_dir)
