import numpy as np
import cv2
import glob


def rotate_longitude(frame, lon):
    h, w, _ = frame.shape
    shift_width = int(w * lon / 360)
    rotated_frame = np.zeros_like(frame)
    if shift_width > 0:
        rest_width = w - shift_width
        rotated_frame[:, 0:shift_width] = frame[:, rest_width:w]
        rotated_frame[:, shift_width:w] = frame[:, 0:rest_width]
    else:
        shift_width *= -1
        rest_width = w - shift_width
        rotated_frame[:, 0:rest_width] = frame[:, shift_width:w]
        rotated_frame[:, rest_width:w] = frame[:, 0:shift_width]
    return rotated_frame


def gen_frames(file_path):
    file_name = file_path.split('/')[-1].split('.')[0]
    lat = int(file_name.split('_')[0])
    frame = cv2.imread(file_path)
    for lon in range(-180, 180, 3):
        rotated_frame = rotate_longitude(frame, lon)
        cv2.imwrite('./out/{}_{}.png'.format(lat, 180 + lon), rotated_frame)


def run():
    files = glob.glob("./input/*")
    for file in files:
        gen_frames(file)


if __name__ == '__main__':
    run()
