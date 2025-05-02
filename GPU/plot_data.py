import pandas as pd
import matplotlib.pyplot as plt

# Read data
df_results = pd.read_csv('results.txt', header=None, names=['Time (ms)', 'Voltage (V)'])
df_clean = pd.read_csv('data_clean.txt', header=None, names=['Time (ms)', 'Voltage (V)'])

# Use time from data_clean.txt
time = df_clean['Time (ms)'].to_numpy()

# Get voltages
voltage_results = df_results['Voltage (V)'].to_numpy()
voltage_clean = df_clean['Voltage (V)'].to_numpy()

# Plot
plt.figure(figsize=(8, 4))
plt.plot(time, voltage_clean, label='data_clean.txt', linestyle='--')
plt.plot(time, voltage_results, label='results.txt', linestyle='-')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs. Time Overlay')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
