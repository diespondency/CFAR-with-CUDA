import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import label
from matplotlib.widgets import Slider
import os
import csv

# --- Configuration ---
fs = 250e3  # Sample rate
sim_files = ['received_data_1.csv', 'received_data_2.csv', 'received_data_3.csv', 'received_data_4.csv']

rir_data = []

# --- CFAR Function ---
def cfar(X_k, num_guard_cells, num_ref_cells, bias_plus, bias_multiplier):
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
        output = mean * bias_multiplier + bias_plus
        cfar_values[center_index] = output

    targets_only = np.copy(X_k)
    targets_only[X_k < cfar_values] = np.nan
    return cfar_values, targets_only

# --- Load Simulation Data ---
for filename in sim_files:

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        sim_data = np.array([[float(x[0]), float(x[1])] for x in reader])

    sim_time, sim_values = sim_data[:, 0], sim_data[:, 1]
    sim_values = np.abs(sim_values)  # Ensure all values are positive
    sim_values = sim_values - np.mean(sim_values)  # Remove DC offset
    rirSim = sim_values
    t = sim_time[0:6250]
    filtered_signal = savgol_filter(rirSim, 101, 2)
    #filtered_signal = sim_values[0:6250]  # Use the first 6250 samples directly
    rir_data.append({
        "filename": filename,
        "t": t,
        "filtered_signal": filtered_signal,
    })

# --- Initial CFAR Parameters ---
initial_bias_plus = 4.36
initial_bias_multiplier = 3.53
initial_guard_cells = 65
initial_ref_cells = 40

fig, axes = plt.subplots(len(rir_data), 1, figsize=(12, 2.5 * len(rir_data)), sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.4)
if len(rir_data) == 1:
    axes = [axes]

plot_lines = []

# --- Initial Plot ---
for idx, data in enumerate(rir_data):
    ax = axes[idx]
    t = data["t"]
    filtered_signal = data["filtered_signal"]

    threshold, targets_only = cfar(filtered_signal, initial_guard_cells, initial_ref_cells, initial_bias_plus, initial_bias_multiplier)

    line_signal, = ax.plot(t, filtered_signal, label="Filtered", color="blue")
    line_thresh, = ax.plot(t, threshold, label="CFAR Thresh", color="orange")
    targets_scatter = ax.scatter(t, targets_only, color="green", label="Targets", marker='x')
    ax.set_title(f"RIR {data['filename']}")
    ax.legend()
    ax.grid(True)

    plot_lines.append((line_thresh, targets_scatter))

# --- Slider Setup ---
ax_bias_plus = plt.axes([0.1, 0.14, 0.65, 0.03])
ax_bias_multiplier = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_guard = plt.axes([0.1, 0.06, 0.65, 0.03])
ax_ref = plt.axes([0.1, 0.02, 0.65, 0.03])


slider_bias_plus = Slider(ax_bias_plus, 'Bias+', 1, 20, valinit=initial_bias_plus)
slider_bias_multiplier = Slider(ax_bias_multiplier, 'Bias Multiplier', 1, 20, valinit=initial_bias_multiplier)
slider_guard = Slider(ax_guard, 'Guard Cells', 0, 100, valinit=initial_guard_cells, valstep=1)
slider_ref = Slider(ax_ref, 'Ref Cells', 0, 100, valinit=initial_ref_cells, valstep=1)

# --- Update Function for Sliders ---
def update(val):
    bias_plus = slider_bias_plus.val
    bias_multiplier = slider_bias_multiplier.val
    num_guard_cells = int(slider_guard.val)
    num_ref_cells = int(slider_ref.val)

    for idx, data in enumerate(rir_data):
        t = data["t"]
        signal = data["filtered_signal"]
        threshold, targets_only = cfar(signal, num_guard_cells, num_ref_cells, bias_plus, bias_multiplier)

        line_thresh, targets_scatter = plot_lines[idx]
        line_thresh.set_ydata(threshold)

        target_mask = ~np.isnan(targets_only)
        labeled_array, num_features = label(target_mask)

        mean_times = []
        mean_amps = []

        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            mean_time = np.mean(t[indices])
            mean_amp = np.mean(signal[indices])
            mean_times.append(mean_time)
            mean_amps.append(mean_amp)

        targets_scatter.set_offsets(np.column_stack((mean_times, mean_amps)))

    fig.canvas.draw_idle()

slider_bias_plus.on_changed(update)
slider_bias_multiplier.on_changed(update)
slider_guard.on_changed(update)
slider_ref.on_changed(update)
plt.show()
