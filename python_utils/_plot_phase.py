import os
import numpy as np
import matplotlib.pyplot as plt

TYPE = ["raw", "unwrapped", "detrended"]
DIR_NAME = "experiment/phase"

NUM_CLASS = 10
NUM_SUB_CARR = 128
LINES_PER_CLASS = 690


def main():
    for file_type in TYPE:
        print(f"Processing {file_type}...")

        data = np.loadtxt(f"{file_type}_phase.txt")

        for sc in range(NUM_SUB_CARR):
            sc_dir = f"{DIR_NAME}/{file_type}/sc_{sc+1}"
            os.makedirs(sc_dir, exist_ok=True)

            for cls in range(NUM_CLASS):
                start = cls * LINES_PER_CLASS
                end = (cls + 1) * LINES_PER_CLASS

                # Extract one subcarrier for one class
                subcarrier_data = data[start:end, sc]

                # Save TXT
                np.savetxt(
                    f"{sc_dir}/class_{cls}_{file_type}.txt",
                    subcarrier_data,
                    fmt="%.6f"
                )

                # Plot PNG
                # plt.figure(figsize=(18, 5))
                # plt.plot(subcarrier_data)

                # plt.title(
                #     f"Class {cls} - Subcarrier {sc+1} - {file_type.capitalize()} Phase"
                # )
                # plt.xlabel("Sample Index")
                # plt.ylabel("Phase")
                # plt.grid(True)

                # # Save plot
                # plt.savefig(
                #     f"{sc_dir}/class_{cls}_{file_type}.png",
                #     dpi=300,
                #     bbox_inches="tight"
                # )

                # plt.close()   # important: free memory


if __name__ == "__main__":
    main()