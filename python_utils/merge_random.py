import os
import random
import csv

# =========================
# CONFIG
# =========================
BASE_DIR = "rawData"
OUTPUT_FILE = "data.csv"
N = 690  # number of random lines from each folder

all_selected_lines = []

# Loop through folders 0 to 9
for i in range(10):
    folder_path = os.path.join(BASE_DIR, str(i))
    csv_path = os.path.join(folder_path, "h.csv")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        continue

    # Read all lines
    with open(csv_path, "r") as f:
        lines = f.readlines()

    modified_lines = []

    # Replace first item with folder name
    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split(",")

        # Replace first item
        parts[0] = str(i)

        modified_lines.append(parts)

    # Randomly choose N lines
    if len(modified_lines) < N:
        print(f"Folder {i} has only {len(modified_lines)} lines. Taking all.")
        selected = modified_lines
    else:
        selected = random.sample(modified_lines, N)

    # IMPORTANT:
    # Keep order grouped by folder
    # Do NOT shuffle globally
    all_selected_lines.extend(selected)

# Save merged CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(all_selected_lines)

print(f"Saved {len(all_selected_lines)} lines into {OUTPUT_FILE}")