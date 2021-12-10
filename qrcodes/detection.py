import cv2
from pyzbar.pyzbar import decode
import datetime


def conv_y2rad(y: int, height: int):
    return int(90 - y / height * 180)


WriteRaw = True

def run():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    raw_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_fps: int = int(cap.get(cv2.CAP_PROP_FPS))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt_now = datetime.datetime.now()
    writer = cv2.VideoWriter("qr_detection_{}.mp4".format(dt_now.strftime('%Y-%m-%d-%H:%M:%S')), fourcc, video_fps, (width, height))
    raw_writer = cv2.VideoWriter("qr_detection_raw_{}.mp4".format(dt_now.strftime('%Y-%m-%d-%H:%M:%S')), raw_fourcc, video_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if WriteRaw:
            raw_writer.write(frame)

        d = decode(frame)
        if d:
            for qr in d:
                # code = qr.data.decode('utf-8')
                x, y, w, h = qr.rect
                print(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                rad = conv_y2rad(int(y + h / 2), height)
                cv2.putText(frame, str(rad), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255))
                # cv2.putText(frame, code, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255))
                cv2.putText(frame, "w: {}, h: {}".format(w, h), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255))

        writer.write(frame)
        cv2.imshow('view', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            raw_writer.release()
            writer.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    run()
