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
    writer = cv2.VideoWriter("qr_detection_{}.mp4".format(dt_now.strftime('%Y-%m-%d-%H:%M:%S')), fourcc, video_fps,
                             (width, height))
    raw_writer = cv2.VideoWriter("qr_detection_raw_{}.mp4".format(dt_now.strftime('%Y-%m-%d-%H:%M:%S')), raw_fourcc,
                                 video_fps, (width, height))

    photo_name = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        d = decode(frame)
        if d:
            for qr in d:
                # code = qr.data.decode('utf-8')
                x, y, w, h = qr.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                rad = conv_y2rad(int(y + h / 2), height)
                cv2.putText(frame, str(rad), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255))
                # cv2.putText(frame, code, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255))
                cv2.putText(frame, "w: {}, h: {}".format(w, h), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255))

        raw_frame = frame.copy()
        if WriteRaw:
            raw_writer.write(raw_frame)
        writer.write(frame)
        cv2.imshow('view', cv2.resize(frame, dsize=(int(width / 2), int(height / 2))))

        key = cv2.waitKey(1)
        if key == ord('s'):
            file_name = '{}.png'.format(photo_name)
            print(file_name)
            photo_name += 1
            cv2.imwrite('./real_out/raw/{}'.format(file_name), raw_frame)
            for i in range(0, 360, 5):
                x = int(i / 360 * width)
                cv2.line(frame, (x, 0), (x, height), (0, 0, 0), thickness=1)
            for j in range(0, 180, 5):
                y = int(j / 180 * height)
                cv2.line(frame, (0, y), (width, y), (0, 0, 0), thickness=1)
            cv2.imwrite('./real_out/result/{}'.format(file_name), frame)
        if key == ord('q'):
            raw_writer.release()
            writer.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    run()
