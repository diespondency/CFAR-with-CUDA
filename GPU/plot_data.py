import pandas as pd
import matplotlib.pyplot as plt

# Read data
# df_results = pd.read_csv('results.txt', header=None, names=['Time (ms)', 'Voltage (V)'])
df = pd.read_csv('detailed_block_size_results.csv')

# Use time from data_clean.txt
# time = df_clean['Time (ms)'].to_numpy()

# Get voltages
# voltage_results = df_results['Voltage (V)'].to_numpy()
# voltage_clean = df_clean['Voltage (V)'].to_numpy()

threads = df['ThreadsPerBlock'].to_numpy()
threads = threads[64::]
avg_time = df['AvgTimeMs'].to_numpy()
avg_time = avg_time[64::]
# Plot
plt.figure(figsize=(8, 4))
plt.plot(threads,avg_time, label='data_clean.txt')
# plt.plot(time, voltage_results, label='results.txt', linestyle='-'
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.grid
plt.title('Voltage vs. Time Overlay')
plt.legend()
plt.tight_layout()
plt.show()
