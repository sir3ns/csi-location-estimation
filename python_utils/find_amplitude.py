import numpy as np
import matplotlib.pyplot as plt


def get_amplitude(csi):
    I = csi[0::2]
    Q = csi[1::2]

    min_len = min(len(I), len(Q))
    I = I[:min_len]
    Q = Q[:min_len]

    amp = np.sqrt(I**2 + Q**2)
    return amp


def parse_csi_line(line):
    data = line.strip().split(",")
    ts = int(data[0])
    vals = np.array(list(map(int, data[1:])))
    amp = get_amplitude(vals)

    return ts, amp


# load file
def load_csi(file_path):
    timestamps = []
    samples = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("timestamp"):
                continue
            ts, amp = parse_csi_line(line)
            timestamps.append(ts)
            samples.append(amp)

    return np.array(timestamps), np.array(samples)


ts, raw = load_csi("./rawData/1/h.csv")

# -------- MEAN AMPLITUDE --------

mean_amp = np.mean(raw, axis=1)

plt.figure(figsize=(12, 5))
plt.plot(ts, mean_amp)

plt.title("Mean CSI Amplitude vs Timestamp")
plt.xlabel("Timestamp")
plt.ylabel("Mean Amplitude")
plt.grid(True)
plt.show()


# -------- SINGLE SUBCARRIER --------

# subcarrier_index = 0
# sub_amp = raw[:, subcarrier_index]

# plt.figure(figsize=(12, 5))
# plt.plot(ts, sub_amp)

# plt.title(f"CSI Subcarrier {subcarrier_index} vs Timestamp")
# plt.xlabel("Timestamp")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()