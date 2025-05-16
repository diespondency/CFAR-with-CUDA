import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import os
import csv
from scipy.signal import chirp

fs = 250e3  # Sample rate

# List of CSV filenames
csv_files = ['received_data_1.csv', 'received_data_2.csv', 'received_data_3.csv', 'received_data_4.csv']

# Prepare subplots
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 15), sharex=True)

for i, filename in enumerate(csv_files):
    time = []
    voltage = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            time.append(float(row[0]))
            voltage.append(float(row[1]))
    
    voltage = np.array(voltage)
    voltage = voltage * 0.01
    
    axs[i].plot(time, voltage, label=f'{filename}')
    axs[i].set_title(f'Data from {filename}')
    axs[i].set_ylabel('Voltage (V)')
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('received_data_plots.png', dpi=600)
plt.show()