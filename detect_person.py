from typing import Tuple, List
import cv2
import numpy as np
import darknet


def convert2relative(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x / _width, y / _height, w / _width, h / _height


def convert2original(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = convert2relative(bbox)
    image_h, image_w, __ = image.shape
    return int(x * image_w), int(y * image_h), int(w * image_w), int(h * image_h)


# 引数は検出中心の(x, y, w, h）
def convert_xywh2tlwh(xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return int(xywh[0] - xywh[2] / 2), int(xywh[1] - xywh[3] / 2), xywh[2], xywh[3]


network, class_names, class_colors = darknet.load_network(
    "./cfg/yolov4.cfg",
    "./cfg/coco.data",
    "./yolov4.weights",
    batch_size=1
)
darknet_width: int = darknet.network_width(network)
darknet_height: int = darknet.network_height(network)

img = cv2.imread('over_zero.jpg')
height, width, _ = img.shape


frame_rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
frame_resized: np.ndarray = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
darknet_detections = darknet.detect_image(network, class_names, darknet_image, thresh=.25)
darknet.free_image(darknet_image)
for label, confidence, xywh in darknet_detections:
    if label != "person" or float(confidence) < 80:
        continue
    bbox_adjusted = convert2original(img, xywh)
    tlwh = convert_xywh2tlwh(bbox_adjusted)
    cv2.rectangle(img, tlwh[:2], (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]), (255, 0, 0), 20)

cv2.imwrite('person_in_normal.jpg', img)
