import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Constants
speed_of_sound = 343  # Speed of sound in air in m/s
distance_to_wall = 4  # Distance to the wall in meters
sampling_rate = 250000  # Sampling rate in Hz
pulse_duration = 0.001  # Duration of the pulse in seconds

# Time array
t = np.linspace(0, 4 * distance_to_wall / speed_of_sound, int(4 * distance_to_wall / speed_of_sound * sampling_rate))

# Generate pulse
pulse = np.zeros_like(t)
pulse_start = int(0.015 * sampling_rate)
pulse_end = pulse_start + int(pulse_duration * sampling_rate)
pulse[pulse_start:pulse_end] = 1

# Simulate received signal
received_signal = np.zeros_like(t)
reflection_start = int((2 * distance_to_wall / speed_of_sound) * sampling_rate) + pulse_start
received_signal[reflection_start:reflection_start + (pulse_end - pulse_start)] = 0.5

# Combine pulse and received signal
totSig = pulse + received_signal

# Design low-pass filter
cutoff_frequency = 1000  # Cutoff frequency in Hz
nyquist_rate = sampling_rate / 2
normal_cutoff = cutoff_frequency / nyquist_rate
b, a = butter(1, normal_cutoff, btype='low', analog=False)

# Apply low-pass filter
filtered_signal = filtfilt(b, a, totSig)

# Add white Gaussian noise
noise = np.random.normal(0, 0.15, totSig.shape)  # Adjust the standard deviation for noise level
filtered_signal = filtered_signal + noise

# Save filtered_signal to CSV file
np.savetxt("filtered_signal.csv", filtered_signal, delimiter=",")

print(filtered_signal)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, totSig, label='Noisy Signal', color="orange")
plt.plot(t, filtered_signal, label='Envelope', color="blue")
plt.scatter(t, filtered_signal, color='red', label='Sampled Envelope')
# Add vertical lines at each x-value
#for x in t:
    #plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Acoustic Pulse Transmission and Reception with Noise and Envelope')
plt.legend()
plt.grid(True)
plt.show()