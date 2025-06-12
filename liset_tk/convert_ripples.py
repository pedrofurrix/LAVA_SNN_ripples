import numpy as np
import pandas as pd
import os
# Load the txt file
abspath=r"C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\Download_from_paper\PV01FPGA241003_162924"
ripple_path=os.path.join(abspath, "events_selected_manually.txt")
ripples = np.loadtxt(ripple_path)  # Update this if your file has a different name/path

# Multiply by 30000 (convert seconds to samples)
ripples_samples = np.round(ripples * 30000).astype(int)

# Create a DataFrame with proper column names
df = pd.DataFrame(ripples_samples, columns=["ripIni", "ripEnd"])

# Save to CSV
csv_path = os.path.join(abspath, "ripples.csv")
df.to_csv(csv_path, index=False,sep=",")