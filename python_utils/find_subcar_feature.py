import os
import matplotlib.pyplot as plt

folder_path = './data.csv'

with open(folder_path, "r") as f:
    lines = f.readlines()

def save_plot(sub_car_amp):
    all_amp = []

    for line in lines:
        line = line.strip().split(',')
        I = int(line[sub_car_amp * 2 - 1])
        Q = int(line[sub_car_amp * 2])
        amp = (I**2 + Q**2)**0.5
        all_amp.append(amp)
        # break

    plt.figure(figsize=(20, 9))
    plt.plot(range(6900), all_amp)

    plt.title(f'Amplitude {sub_car_amp}')
    plt.xlabel('Sample')
    plt.ylabel('Value')

    plt.savefig(f'{sub_car_amp}.png')
    plt.close()


for i in range(1, 129):
    save_plot(i)