import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
import csv
import time
import math
from scipy.signal import chirp, convolve

# Load configuration
with open('config.json') as json_file:
    config = json.load(json_file)

def tx_rx_NIDAQ(config):
    sample_rate = config["sample_rate"]
    ##T_sweep = config["T_sweep"]
    T_sweep = config["n_meas"] / sample_rate # Sweep duration (seconds)
    dev_id = config["id_daq"]
    f1 = config["f1"]
    f2 = config["f2"]

    ##t = np.linspace(0, T_sweep, int(sample_rate * T_sweep), endpoint=False)
    t = np.linspace(0, config["n_meas"]/sample_rate, config["n_meas"], endpoint=False)
    signal = config["amp"] * chirp(t, f0=f1, f1=f2, t1=T_sweep, method='log')
    
    time_axis, data = read_csv_data('received_data.csv') ##remove if testing  

    print("Signal length:", len(signal))
    print("Data length:", len(data))

    if len(signal) != len(data):
        raise ValueError("Signal length and Data length do not match!")
    
    return time_axis, data, signal ##remove if testing  

    # The sweep itself is not the target — the response to it is.
    # You need to capture all the direct sound + reflections in time-alignment with the original signal.
    
    Task_I = nidaqmx.Task()
    Task_O = nidaqmx.Task()

    Task_I.ai_channels.add_ai_voltage_chan(dev_id + '/' + config["id_input"],
                                           min_val=config["min_val_in"],
                                           max_val=config["max_val_in"],
                                           terminal_config=TerminalConfiguration.RSE)

    Task_I.timing.cfg_samp_clk_timing(rate=sample_rate,
                                      source='ao/SampleClock',
                                      sample_mode=AcquisitionType.FINITE,
                                      samps_per_chan=config["n_meas"])

    Task_O.ao_channels.add_ao_voltage_chan(dev_id + '/' + config["id_output"])

    Task_O.timing.cfg_samp_clk_timing(rate=sample_rate,
                                      sample_mode=AcquisitionType.FINITE,
                                      samps_per_chan=config["n_meas"])

    Task_O.write(signal, auto_start=False)

    Task_I.start()
    Task_O.start()

    data = Task_I.read(number_of_samples_per_channel=config["n_meas"])
    data = np.array(data)

    Task_I.stop()
    Task_I.close()
    Task_O.stop()
    Task_O.close()

    time_axis = np.linspace(0, (config["n_meas"] / sample_rate), config["n_meas"], endpoint=False)

    return time_axis, data, signal

def calculateInvertedFilter(signal, sample_rate, f1, f2, T_sweep, amp):
    f = np.zeros(int(sample_rate * T_sweep))

    for i in range(len(signal)):
        p = i / sample_rate
        f[i] = amp * signal[int(sample_rate * T_sweep)-1-i] * (2*math.pi*f1) / ((2*math.pi*f2) * math.exp(p/(T_sweep/math.log(f2/f1))))
    return f

def plot_received_signal(config):
    time_axis, data, signal = tx_rx_NIDAQ(config)
    
    # Calculate inverted filter
    f = calculateInvertedFilter(
        signal,
        sample_rate=config["sample_rate"],
        f1=config["f1"],
        f2=config["f2"],
        T_sweep=config["n_meas"] / config["sample_rate"],
        amp=config["amp"]
    )

    # Convolve to deconvolve
    rir_measured_con = convolve(data, f, mode="full")
    rir_measured_con = rir_measured_con[len(signal):(len(signal)+config["n_meas"])]

    plt.figure(figsize=(12, 5))
    plt.subplot(3, 1, 1)
    plt.plot(signal, label="Sine Sweep", color="blue")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(data, label="Data", color="blue")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(f, label="inv Sine Sweep", color="red")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(rir_measured_con, label="Measured signal", color="green")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def read_csv_data(filename):
    time_axis = []
    data = []
    
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            time_axis.append(float(row[0]))  # Time (ms)
            data.append(float(row[1]))       # Voltage (V)
    
    return np.array(time_axis), np.array(data)

# Run the full pipeline
plot_received_signal(config)
