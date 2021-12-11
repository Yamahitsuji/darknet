import numpy as np
import qrcode
import cv2


def run():
    width = 3840
    height = 1920
    code = 12345

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = 255
    qr = qrcode.make(str(code), version=1, box_size=12, border=0,
                     error_correction=qrcode.constants.ERROR_CORRECT_H).convert('RGB')
    qr_img = cv2.resize(np.array(qr, dtype=np.uint8), dsize=(70, 70))
    qr_height, qr_width, _ = qr_img.shape
    top = int(height / 2 - qr_height / 2)
    bottom = top + qr_height
    left = int(width / 2 - qr_width / 2)
    right = left + qr_width
    frame[top:bottom, left:right] = qr_img

    cv2.imwrite('out/0_0.png', frame)


if __name__ == '__main__':
    run()
