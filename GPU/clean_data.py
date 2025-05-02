import numpy as np

# Read the data from 'data.txt'
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Split each line into x and y values
x_vals = []
y_vals = []
for line in lines:
    parts = line.strip().split(',')
    if len(parts) == 2:
        x, y = map(float, parts)
        x_vals.append(x)
        y_vals.append(y)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

# Remove the DC component (subtract the mean from y values)
mean_value = np.mean(y_vals)
y_vals_no_dc = y_vals - mean_value

# Take the absolute value
y_vals_clean = np.abs(y_vals_no_dc)

# Write the cleaned data to 'data_clean.txt', keeping the original x values
with open('data_clean.txt', 'w') as file:
    for x, y_clean in zip(x_vals, y_vals_clean):
        file.write(f"{x},{y_clean}\n")

print("Data processing complete. Cleaned data saved to 'data_clean.txt'.")
