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


def has_contrast_detection(tlwh: Tuple[int, int, int, int], candidates: List[Tuple[int, int, int, int]], img_width: int,
                           threshold: float = 0.7) -> bool:
    shifted_tl = (tlwh[0] + img_width, tlwh[1])
    shifted_br = (shifted_tl[0] + tlwh[2], shifted_tl[1] + tlwh[3])
    candidates: np.ndarray = np.array(candidates)
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(shifted_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(shifted_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(shifted_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(shifted_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = tlwh[2] * tlwh[3]
    area_candidates = candidates[:, 2:].prod(axis=1)
    ious = area_intersection / (area_bbox + area_candidates - area_intersection)
    return ious.max() >= threshold


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
padding_width: int = int(width / 4)
expanded_width: int = width + padding_width * 2
quarter_left: int = int(padding_width + width / 4)

expanded_frame: np.ndarray = np.concatenate(
    [img[:, int(width * 3 / 4):], img, img[:, :int(width / 4)]], 1)
frame_rgb: np.ndarray = cv2.cvtColor(expanded_frame, cv2.COLOR_BGR2RGB)
frame_resized: np.ndarray = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
darknet_detections = darknet.detect_image(network, class_names, darknet_image, thresh=.25)
darknet.free_image(darknet_image)
adjusted_detections = []
for label, confidence, xywh in darknet_detections:
    if label != "person" or float(confidence) < 80:
        continue
    bbox_adjusted = convert2original(expanded_frame, xywh)
    tlwh = convert_xywh2tlwh(bbox_adjusted)
    adjusted_detections.append((label, confidence, tlwh))


# 検出の重複を削除する
# 1. personのみにする
# 2. 元画像の範囲に存在する検出に絞る
# 3. 左右の重複を削除
detections_in_area: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
candidates_tlwh: List[Tuple[int, int, int, int]] = []
expanded_frame_all_person = expanded_frame.copy()
for (label, confidence, tlwh) in adjusted_detections:
    if tlwh[0] + tlwh[2] < padding_width or tlwh[0] >= padding_width + width:
        continue
    # 全ユーザの検出結果
    cv2.rectangle(expanded_frame_all_person, tlwh[:2], (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]), (255, 0, 0), 20)
    detections_in_area.append((label, confidence, tlwh))
    candidates_tlwh.append(tlwh)
cv2.line(expanded_frame_all_person, (padding_width, 0), (padding_width, height), (0, 0, 255), thickness=20)
cv2.line(expanded_frame_all_person, (padding_width + width - 1, 0), (padding_width + width - 1, height),
         (0, 0, 255), thickness=20)

for (label, confidence, tlwh) in detections_in_area:
    if tlwh[0] + tlwh[2] < quarter_left:
        continue
    cv2.rectangle(expanded_frame, tlwh[:2], (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]), (255, 0, 0), 20)

cv2.line(expanded_frame, (padding_width, 0), (padding_width, height), (0, 0, 255), thickness=20)
cv2.line(expanded_frame, (padding_width + width - 1, 0), (padding_width + width - 1, height),
         (0, 0, 255), thickness=20)

cv2.imwrite('all_person.jpg', expanded_frame_all_person)
cv2.imwrite('valid_person.jpg', expanded_frame)
