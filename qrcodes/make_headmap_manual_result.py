import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

f = open('result_count.txt', 'r')
datalist = f.readlines()
f.close()

total = np.zeros((18, 36), dtype=np.int)
detected = np.zeros((18, 36), dtype=np.int)
for line in datalist:
    filename, result = line.split()
    lat_str, lon_str, flag = result.split('-')
    lat = int(lat_str)
    lon = int(lon_str)
    can_read = flag == '1'
    total[lat, lon] += 1
    if can_read:
        detected[lat, lon] += 1

total_for_division = np.where(total == 0, 1, total)
heatmap = detected / total_for_division

# https://sabopy.com/py/matplotlib-55/
x = np.arange(0, 360)
y = np.arange(-90, 90)
mesh = np.meshgrid(x, y)
resized = cv2.resize(heatmap, (360, 180))
plt.rcParams["font.size"] = 15
plt.contourf(mesh[0], mesh[1], resized, cmap='Spectral_r')
plt.xlabel("longitude")
plt.xlim(0, 360)
plt.ylabel("latitude")
plt.ylim(-90, 90)
cbar = plt.colorbar()
cbar.ax.set_ylim(0, 1)
plt.savefig('./graph/read_qr_heatmap.png')
