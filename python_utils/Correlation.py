import numpy as np
import matplotlib.pyplot as plt

# ===== LOAD DATA =====

with open("data.csv", "r") as f:
    lines = f.readlines()

# ===== EXTRACT AMPLITUDES =====

all_samples = []

for line in lines:

    vals = list(map(int, line.strip().split(',')))

    label = vals[0]
    iq = vals[1:]

    amplitudes = []

    for i in range(0, len(iq), 2):

        I = iq[i]
        Q = iq[i + 1]

        amp = np.sqrt(I**2 + Q**2)

        amplitudes.append(amp)

    all_samples.append(amplitudes)

# shape = (6900, 128)
X = np.array(all_samples)

# ===== TRANSPOSE =====
# Now shape becomes:
# (128, 6900)

X = X.T

# ===== CORRELATION MATRIX =====

corr_matrix = np.corrcoef(X)

# ===== PLOT =====

plt.figure(figsize=(12, 10))

plt.imshow(corr_matrix)

plt.colorbar(label='Correlation')

plt.title("Subcarrier Correlation Matrix")

plt.xlabel("Subcarrier")
plt.ylabel("Subcarrier")

plt.show()