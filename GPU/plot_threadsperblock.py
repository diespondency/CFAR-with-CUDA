import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('detailed_block_size_results.csv')

# Extract and slice
threads = df['ThreadsPerBlock'].to_numpy()[64:]
avg_time = df['AvgTimeMs'].to_numpy()[64:]

# Create figure + axes
fig, ax = plt.subplots(figsize=(8, 4))

# Plot with markers
ax.plot(threads, avg_time, label='AvgTimeMs')

# Grid
ax.grid(True, which='major', linestyle=':', linewidth=0.5)

# Ticks: one per thread-count on x, ~5 on y
ax.locator_params(axis='x', nbins=40)
ax.locator_params(axis='y', nbins=5)

# Correct labels & title
ax.set_xlabel('Threads Per Block', fontsize=24)
ax.set_ylabel('Average Time (ms)', fontsize=24)
ax.set_title('Average kernel computing time vs. threads per block, averaged over 1000 iterations (Nvidia Geforce GTX 970)', fontsize=24)

ax.tick_params(axis='both', which='major', labelsize=18)
fig.tight_layout()

plt.show()
