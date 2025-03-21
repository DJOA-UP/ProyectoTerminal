import wfdb
import numpy as np
import matplotlib.pyplot as plt

def read_signal(file_path):
    record = wfdb.rdrecord(file_path)  # Reads both .hea and .dat automatically
    signals = record.p_signal  # Extracts signal values as a NumPy array
    return signals, record.fs  # Also return sampling frequency

# Example usage
file_path = r'C:\Users\DJOA0\OneDrive\Desktop\non-eeg-dataset-for-assessment-of-neurological-status-1.0.0\Subject2_SpO2HR'  # Update with actual file path
signals, fs = read_signal(file_path)

# Print shape
print("Signal Shape:", signals.shape)  # (samples, channels)

# Plot signals
time = np.arange(signals.shape[0]) / fs  # Create time axis

plt.figure(figsize=(12, 6))
for i in range(signals.shape[1]):  # Iterate through all channels
    plt.plot(time, signals[:, i], label=f"Channel {i+1}")

plt.xlabel("Time (seconds)")
plt.ylabel("Signal Value")
plt.title("Physiological Signal Data")
plt.legend()
plt.show()
