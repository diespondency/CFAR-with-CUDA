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

# ==================================== TO DO ==================================== #
# test the new def-functions for correct working (also implemented amp for the inverting sweep)
# =============================================================================== #
# Load configuration
with open('config.json') as json_file:
    config = json.load(json_file)

def tx_rx_NIDAQ(config, output_csv='received_data.csv'):
    sample_rate = config["sample_rate"]
    T_sweep = config["T_sweep"]
    dev_id = config["id_daq"]
    f1 = config["f1"]
    f2 = config["f2"]

    t = np.linspace(0, T_sweep, int(sample_rate * T_sweep), endpoint=False)
    signal = config["amp"] * chirp(t, f0=f1, f1=f2, t1=T_sweep, method='log')
    
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

    time_axis = np.linspace(0, (config["n_meas"] / sample_rate)*1000, config["n_meas"])

    return time_axis, data, signal

def calculateInvertedFilter(signal):
    f = np.zeros(int(config["sample_rate"] * config["T_sweep"]))
    for i in range(len(signal)):
        p = i/config["sample_rate"]
        f[i] = config["amp"] * signal[int(config["sample_rate"] * config["T_sweep"])-1-i] * (2*math.pi*config["f1"]) / ((2*math.pi*config["f2"]) * math.exp(p/(config["T_sweep"]/math.log(config["f2"]/config["f1"]))))
    return f

def plot_received_signal(config):
    time_axis, data, signal = tx_rx_NIDAQ(config)
    f = calculateInvertedFilter(signal)

    rir_measured_con = convolve(data, f, mode="full")
    rir_measured_con = rir_measured_con[len(signal):(len(signal)+config["n_meas"])]
    
    with open("received_data.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (ms)", "Voltage (V)"])
        writer.writerows(zip(time_axis[:len(rir_measured_con)], rir_measured_con))

    plt.figure(figsize=(10, 10))
    plt.plot(time_axis[:len(rir_measured_con)], rir_measured_con, label="Received Signal", color='b')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.title("Real-Time Received Signal (40 kHz Block Pulse)")
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
            time_axis.append(float(row[0]))  # First column: Time (ms)
            data.append(float(row[1]))  # Second column: Voltage (V)
    
    return np.array(time_axis), np.array(data)

# Call the function with your config
plot_received_signal(config)
    
