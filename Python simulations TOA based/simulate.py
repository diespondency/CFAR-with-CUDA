import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import csv
from scipy.ndimage import label
from scipy.signal import savgol_filter

# Constants
sampling_rate = 250000
cutoff_frequency = 1000
nyquist_rate = sampling_rate / 2
normal_cutoff = cutoff_frequency / nyquist_rate

# Read CSV Data
def read_csv_data(filename):
    time_axis = []
    data = []
    
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            time_axis.append(float(row[0]))  # First column: Time (s)
            data.append(float(row[1]))  # Second column: Voltage (V)
    
    return np.array(time_axis), np.array(data)

time_axis, data = read_csv_data("dataMIC0.csv")

# Design low-pass filter
filtered_signal = savgol_filter(data, 101, 2)

# CFAR function
def cfar(X_k, num_guard_cells, num_ref_cells, bias):
    N = len(X_k)
    cfar_values = np.zeros(N)
    
    for center_index in range(num_guard_cells + num_ref_cells, N - (num_guard_cells + num_ref_cells)):
        min_index = center_index - (num_guard_cells + num_ref_cells)
        min_guard = center_index - num_guard_cells
        max_index = center_index + (num_guard_cells + num_ref_cells) + 1
        max_guard = center_index + num_guard_cells + 1

        lower_nearby = X_k[min_index:min_guard]
        upper_nearby = X_k[max_guard:max_index]
        mean = np.mean(np.concatenate((lower_nearby, upper_nearby)))
        output = mean * bias
        cfar_values[center_index] = output

    targets_only = np.copy(X_k)
    targets_only[X_k < cfar_values] = np.nan  # Mask values below threshold
    return cfar_values, targets_only

# Initial CFAR parameters
initial_bias = 9
initial_guard_cells = 100
initial_ref_cells = 500
filtered_signal = np.abs(filtered_signal)

threshold, targets_only = cfar(filtered_signal, initial_guard_cells, initial_ref_cells, initial_bias)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

filtered_line, = ax.plot(time_axis, filtered_signal, label='Filtered Signal', color='blue')
thresh_line, = ax.plot(time_axis, threshold, label='CFAR Threshold', color='yellow')
targets_scatter = ax.scatter(time_axis, targets_only, color='green', label='Detected Targets', marker='x')


ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('CFAR Detection with Adjustable Parameters')
ax.legend()
ax.grid(True)

# Slider axes
ax_bias = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_guard = plt.axes([0.1, 0.06, 0.65, 0.03])
ax_ref = plt.axes([0.1, 0.02, 0.65, 0.03])

# Sliders
slider_bias = Slider(ax_bias, 'Bias', 1, 20, valinit=initial_bias)
slider_guard = Slider(ax_guard, 'Guard Cells', 10, 200, valinit=initial_guard_cells, valstep=1)
slider_ref = Slider(ax_ref, 'Ref Cells', 0, 5000, valinit=initial_ref_cells, valstep=10)

# Update function
def update(val):
    bias = slider_bias.val
    num_guard_cells = int(slider_guard.val)
    num_ref_cells = int(slider_ref.val)

    threshold, targets_only = cfar(filtered_signal, num_guard_cells, num_ref_cells, bias)

    thresh_line.set_ydata(threshold)
    target_mask = ~np.isnan(targets_only)
    labeled_array, num_features = label(target_mask)

    mean_times = []
    mean_amps = []

    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        mean_time = np.mean(time_axis[indices])
        mean_amp = np.mean(filtered_signal[indices])
        mean_times.append(mean_time)
        mean_amps.append(mean_amp)

    mean_times = np.array(mean_times)
    mean_amps = np.array(mean_amps)

    targets_scatter.set_offsets(np.column_stack((mean_times, mean_amps)))

    fig.canvas.draw_idle()

slider_bias.on_changed(update)
slider_guard.on_changed(update)
slider_ref.on_changed(update)

plt.show()
