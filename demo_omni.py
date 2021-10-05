from typing import Tuple, List
import cv2
import numpy as np
import math
from timeit import time

from omni_tracking.detection import Detection
from omni_tracking.tracker import Tracker
from omni_tracking import nn_matching
from omni_tracking.tools import generate_detections as gdet
from omni_tracking.tools import preprocessing

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

cap = cv2.VideoCapture("./omni_people.mp4")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_fps: int = int(cap.get(cv2.CAP_PROP_FPS))
width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
padding_width: int = int(width / 4)
expanded_width: int = width + padding_width * 2
quarter_left: int = int(padding_width + expanded_width / 4)
writer = cv2.VideoWriter("out1.mp4", fourcc, video_fps, (expanded_width, height))

# tracking parameters
max_cosine_distance = 0.3
model_filename = 'deep_sort_yolov4/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(max_cosine_distance)
tracker = Tracker(width, height, metric)
nms_max_overlap = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t1 = time.time()

    expanded_frame: np.ndarray = np.concatenate(
        [frame[:, int(width * 3 / 4):], frame, frame[:, :int(width / 4)]], 1)
    frame_rgb: np.ndarray = cv2.cvtColor(expanded_frame, cv2.COLOR_BGR2RGB)
    frame_resized: np.ndarray = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    darknet_detections = darknet.detect_image(network, class_names, darknet_image, thresh=.25)
    darknet.free_image(darknet_image)

    adjusted_detections = []
    for label, confidence, xywh in darknet_detections:
        bbox_adjusted = convert2original(expanded_frame, xywh)
        tlwh = convert_xywh2tlwh(bbox_adjusted)
        adjusted_detections.append((label, confidence, tlwh))

    # Run non-maxima suppression.
    boxes = np.array([d[2] for d in adjusted_detections])
    scores = np.array([d[1] for d in adjusted_detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    adjusted_detections = [adjusted_detections[i] for i in indices]

    # 検出の重複を削除する
    # 1. personのみにする
    # 2. 元画像の範囲に存在する検出に絞る
    # 3. 左右の重複を削除
    detections_in_area: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    candidates_tlwh: List[Tuple[int, int, int, int]] = []
    for (label, confidence, tlwh) in adjusted_detections:
        # TODO: 検出でpersonのみにしたい
        if label != "person":
            continue
        if tlwh[0] + tlwh[2] < padding_width or tlwh[0] >= padding_width + width:
            continue
        detections_in_area.append((label, confidence, tlwh))
        candidates_tlwh.append(tlwh)
    valid_detections: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for (label, confidence, tlwh) in detections_in_area:
        if tlwh[0] + tlwh[2] < quarter_left and has_contrast_detection(tlwh, candidates_tlwh, width):
            break
        valid_detections.append((label, confidence, tlwh))

    detections: List[Detection] = []
    features = encoder(expanded_frame, [detection[2] for detection in valid_detections])
    for detection, feature in zip(valid_detections, features):
        confidence = detection[1]
        tlwh = detection[2]
        theta = (tlwh[0] - padding_width) / width * math.tau
        phi = (tlwh[1] - height / 2) / (height / 2) * math.pi / 2
        radius = int(width/math.tau)
        detections.append(Detection(theta, phi, radius, tlwh[2], tlwh[3], confidence, feature, width, height))

    tracker.predict()
    tracker.update(detections)
    for track in tracker.tracks:
        bbox_color: Tuple[int, int, int] = (255, 0, 0)  # track is confirmed
        if track.is_tentative():
            bbox_color = (70, 200, 70)
        if track.is_deleted():
            bbox_color = (80, 80, 255)
        tlbr = track.to_tlbr()
        tlbr = (tlbr[0] + padding_width, tlbr[1], tlbr[2] + padding_width, tlbr[3])
        cv2.rectangle(expanded_frame, tlbr[:2], tlbr[2:], bbox_color, 3)
        cv2.putText(expanded_frame, "ID: {0}".format(track.track_id), (tlbr[0], tlbr[1]), 0, 1.5e-3 * frame.shape[0],
                    bbox_color, 3)

    cv2.line(expanded_frame, (padding_width, 0), (padding_width, height), (0, 0, 255), thickness=5)
    cv2.line(expanded_frame, (padding_width + width - 1, 0), (padding_width + width - 1, height),
             (0, 0, 255), thickness=5)
    writer.write(expanded_frame)
    cv2.imshow('Inference', expanded_frame)
    cv2.waitKey(1)

cap.release()
writer.release()
cv2.destroyAllWindows()
