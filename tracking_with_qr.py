from typing import Tuple, List
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import math
import qrcode

from omni_tracking.detection import Detection
from omni_tracking.tracker import Tracker
from omni_tracking import nn_matching
from omni_tracking.tools import generate_detections as gdet
from omni_tracking.tools import preprocessing

import darknet

STATUS_INIT = 0
STATUS_TRACKED = 1


class User:
    def __init__(self, uid: str):
        self.uid = uid
        self.frame = None
        self.status = STATUS_INIT

    def gen_qr(self):
        img = qrcode.make(self.uid, version=1, box_size=12,
                          error_correction=qrcode.constants.ERROR_CORRECT_H).convert('RGB')
        return np.array(img)


class UserCollection:
    def __init__(self):
        self.users = []

    def get_user_by_id(self, uid) -> User:
        for user in self.users:
            if user.uid == uid:
                return user

    def add_user(self, user: User):
        self.users.append(user)


user_collection = UserCollection()


# detection and tracking
def tracking():
    network, class_names, class_colors = darknet.load_network(
        "./cfg/yolov4.cfg",
        "./cfg/coco.data",
        "./yolov4.weights",
        batch_size=1
    )
    darknet_width: int = darknet.network_width(network)
    darknet_height: int = darknet.network_height(network)

    cap = cv2.VideoCapture('./omni_qr.mp4')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_fps: int = int(cap.get(cv2.CAP_PROP_FPS))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    padding_width: int = int(width / 4)
    radius = int(width / math.tau)
    expanded_width: int = width + padding_width * 2
    quarter_left: int = int(padding_width + expanded_width / 4)
    writer = cv2.VideoWriter("out_tracking_with_qr.mp4", fourcc, video_fps, (expanded_width, height))

    # tracking parameters
    max_cosine_distance = 0.3
    model_filename = 'deep_sort_yolov4/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(max_cosine_distance)
    tracker = Tracker(width, height, metric, max_iou_distance=0.7)
    nms_max_overlap = 1.0

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

    def has_contrast_detection(tlwh: Tuple[int, int, int, int], candidates: List[Tuple[int, int, int, int]],
                               img_width: int,
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

    def convert2rad(x: int, y: int, x_offset: int) -> Tuple[float, float]:
        _theta = (x - x_offset) / width * math.tau
        _phi = (y - height / 2) / (height / 2) * math.pi / 2
        return _theta, _phi

    def convert2xyz(theta: float, phi: float) -> Tuple[int, int, int]:
        x = int(radius * math.cos(phi) * math.cos(theta))
        y = int(radius * math.cos(phi) * math.sin(theta))
        z = int(radius * math.sin(phi))
        return x, y, z

    def get_frame_area_tlbr(center_x: int, center_y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        aspect = 4 / 3
        if h / w > aspect:
            w = int(h / aspect)
        else:
            h = int(w * aspect)
        return int(center_y - h / 2), int(center_x - w / 2), int(center_y + h / 2), int(center_x + w / 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        expanded_frame: np.ndarray = np.concatenate(
            [frame[:, int(width * 3 / 4):], frame, frame[:, :int(width / 4)]], 1)
        frame_rgb: np.ndarray = cv2.cvtColor(expanded_frame, cv2.COLOR_BGR2RGB)
        frame_resized: np.ndarray = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                               interpolation=cv2.INTER_LINEAR)
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
            theta, phi = convert2rad(tlwh[0], tlwh[1], padding_width)
            detections.append(Detection(theta, phi, radius, tlwh[2], tlwh[3], confidence, feature, width, height))

        tracker.predict()
        tracker.update(detections)

        # QRコードとトラックを結びつける
        d = decode(expanded_frame)
        if d:
            for code in d:
                uid = code.data.decode('utf-8')
                user = user_collection.get_user_by_id(uid)
                if not user:
                    user = User(uid)
                    user_collection.add_user(user)
                if user.status == STATUS_TRACKED:
                    continue

                x, y, w, h = code.rect
                qr_theta, qr_phi = convert2rad(x, y, padding_width)
                qr_point = np.array(convert2xyz(qr_theta, qr_phi))
                tmp_distance = 1
                nearest_idx = -1
                for i, t in enumerate(tracker.tracks):
                    if t.user:
                        continue
                    t_point = np.array(t.to_xyz())
                    cos_distance = np.inner(qr_point, t_point) / (np.linalg.norm(qr_point) * np.linalg.norm(t_point))
                    if cos_distance < tmp_distance:
                        nearest_idx = i
                        tmp_distance = cos_distance

                if nearest_idx >= 0:
                    user.status = STATUS_TRACKED
                    tracker.tracks[nearest_idx].user = user
                    cv2.rectangle(expanded_frame, (x, y), (x + w, y + y), (0, 0, 255), 1)

        # view
        for track in tracker.tracks:
            if track.user:
                color = (0, 255, 0)
                xywh = track.to_xywh()
                tlbr = get_frame_area_tlbr(xywh[0] + padding_width, xywh[1], xywh[2], xywh[3])
                cv2.rectangle(expanded_frame, tlbr[:2], tlbr[2:], color, 5)
                cv2.putText(expanded_frame, track.user.uid, tlbr[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, color)
            else:
                color = (255, 0, 0)
                tlbr = track.to_tlbr()
                tlbr = (tlbr[0] + padding_width, tlbr[1], tlbr[2] + padding_width, tlbr[3])
                cv2.rectangle(expanded_frame, tlbr[:2], tlbr[2:], color, 5)


        cv2.line(expanded_frame, (padding_width, 0), (padding_width, height), (0, 0, 255), thickness=5)
        cv2.line(expanded_frame, (padding_width + width - 1, 0), (padding_width + width - 1, height),
                 (0, 0, 255), thickness=5)
        writer.write(expanded_frame)
        cv2.imshow('demo', expanded_frame)
        cv2.waitKey(1)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking()
