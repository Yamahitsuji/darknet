import cv2
import seaborn as sns
import numpy as np
import glob
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt


def get_lat_lon(file_path):
    file_name = file_path.split('/')[-1].split('.')[0]
    lat = int(file_name.split('_')[0])
    lon = int(file_name.split('_')[1])
    return lat, lon


def can_read_qr(file_path):
    frame = cv2.imread(file_path)
    d = decode(frame)
    return bool(d)


def run():
    counts = np.zeros((18, 36), dtype=int)
    read = np.zeros((18, 36), dtype=int)

    files = glob.glob("./out/*")
    for file in files:
        lat, lon = get_lat_lon(file)

        # ヒートマップの範囲から外れる。ほぼ確実に読み込めないので外す。
        if lat == 90:
            continue

        y = int(lat / 10) + 9
        x = int(lon / 10)
        counts[y, x] += 1
        if can_read_qr(file):
            read[y, x] += 1

    heatmap = read / counts
    # plt.figure()
    # sns.heatmap(heatmap, cmap='viridis')
    # plt.savefig('./graph/qr_heatmap.png')
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
    plt.savefig('./graph/qr_heatmap.png')


if __name__ == '__main__':
    run()
