from flask import Flask, request
import csv
import time
import signal
import sys

app = Flask(__name__)

csv_file = open("Train.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow(["timestamp", "csi_data"])

print("📁 CSV file opened: csi_data.csv")

@app.route('/csi', methods=['POST'])
def csi():
    data = request.json

    if not data or "csi" not in data:
        return "Invalid data", 400

    # Epoch timestamp
    timestamp = int(time.time()) # *1000 if want mili sec time

    csi_str = ",".join(map(str, data["csi"]))

    csv_writer.writerow([timestamp, csi_str])
    csv_file.flush()

    print(f"Saved at {timestamp} | len={len(data['csi'])}")

    return "OK"

def shutdown_handler(sig, frame):
    print("\n🛑 Ctrl+C detected. Closing file...")
    csv_file.close()
    print("✅ CSV file closed safely.")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)