import cv2
from pyzbar.pyzbar import decode


def conv_y2rad(y: int, height: int):
    return int(90 - y / height * 180)


def run():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_fps: int = int(cap.get(cv2.CAP_PROP_FPS))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter("qr_detection.mp4", fourcc, video_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        d = decode(frame)
        if d:
            for qr in d:
                code = qr.data.decode('utf-8')
                x, y, w, h = qr.rect
                print(qr.rect)
                print((x, y), (x + w, y + y))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                rad = conv_y2rad(int(y + h / 2), height)
                cv2.putText(frame, str(rad), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
                cv2.putText(frame, code, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))

        writer.write(frame)
        cv2.imshow('view', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            writer.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    run()
