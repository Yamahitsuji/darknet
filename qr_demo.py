import cv2
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture("omni_qr.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_fps: int = int(cap.get(cv2.CAP_PROP_FPS))
width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("qr_out.mp4", fourcc, video_fps, (width, height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    d = decode(frame)
    if d:
        for barcode in d:
            x, y, w, h = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode('utf-8')
            frame = cv2.putText(frame, barcodeData, (x, y - 10), font, .5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
