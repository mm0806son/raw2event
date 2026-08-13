"""
Convert NPY to CSV for sanity check.
"""
import numpy as np
import pandas as pd

# Replace with your CSV file path
csv_file = '14h20/raw2event.csv'
npy_file = '14h20/raw2event.npy'

# Load the CSV into a pandas DataFrame
data = np.load(npy_file)
np.savetxt(csv_file, data, delimiter=',', fmt='%d')


print(f"Saved {npy_file} successfully!")
