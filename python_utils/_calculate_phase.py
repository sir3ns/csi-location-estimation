"""
This Code Calculates Raw, unwrapped and detrend Phase from data.csv file.
And save them in individual files.
"""

import numpy as np
import os
# import matplotlib.pyplot as plt  # uncomment for plotting

FILE = "data.csv"

with open(FILE, "r") as f:
    lines = f.readlines()

def detrend_phase(unwrapped):
    """
    Least-squares linear detrend across all 128 subcarriers.

    Removes the linear phase ramp (sampling time offset) and constant
    phase offset (residual CFO) by fitting a line through every point,
    not just the first/last subcarrier. This is robust to noise on any
    single subcarrier, including edge/null subcarriers.
    """
    n = len(unwrapped)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, unwrapped, 1)
    trend = slope * x + intercept
    return unwrapped - trend


def main():
    n_samples = 6900
    raw_matrix = np.zeros((n_samples, 128))
    unwrapped_matrix = np.zeros((n_samples, 128))
    detrended_matrix = np.zeros((n_samples, 128))

    row = 0
    for line in lines:
        line = line.strip().split(',')
        csi = list(map(int, line[1:]))

        csi = np.array(csi)
        I = csi[0::2][:128]
        Q = csi[1::2][:128]
        raw_phase = np.arctan2(Q, I)
        unwrapped = np.unwrap(raw_phase)
        detrended = detrend_phase(unwrapped)

        raw_matrix[row] = raw_phase
        unwrapped_matrix[row] = unwrapped
        detrended_matrix[row] = detrended
        row += 1

    np.savetxt("raw_phase.txt", raw_matrix, fmt="%.6f")
    np.savetxt("unwrapped_phase.txt", unwrapped_matrix, fmt="%.6f")
    np.savetxt("detrended_phase.txt", detrended_matrix, fmt="%.6f")

main()