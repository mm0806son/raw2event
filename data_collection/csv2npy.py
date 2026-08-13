"""
Convert CSV to NumPy array
"""

import numpy as np
import pandas as pd

# Replace with your CSV file path
csv_file = '14h20/fake_gt.csv'
npy_file = '14h20/fake_gt_new.npy'

# Load the CSV into a pandas DataFrame
df = pd.read_csv(csv_file)

# Convert the DataFrame to a NumPy array
data = df.to_numpy()

# Save the array to an .npy file
np.save(npy_file, data)

print(f"Saved {npy_file} successfully!")
