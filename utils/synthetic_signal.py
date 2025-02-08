import numpy as np
import pandas as pd

# Generate time and amplitude data
num_points = 10000
sampling_rate = 100  # Hz
duration = num_points / sampling_rate  # seconds
time = np.linspace(0, duration, num_points)
amplitude = np.sin(2 * np.pi * 5 * time) + 0.5 * np.random.normal(0, 1, num_points)  # Sine wave + noise

# Create a DataFrame
data = pd.DataFrame({"Time": time, "Amplitude": amplitude})

# Save to a CSV file
data.to_csv("signal.csv", index=False, header=False)