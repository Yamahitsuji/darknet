#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import cv2

from deep_sort_yolov4.deep_sort import nn_matching
from deep_sort_yolov4.deep_sort.detection import Detection
from deep_sort_yolov4.deep_sort.tracker import Tracker
from deep_sort_yolov4.tools import generate_detections as gdet
import imutils.video

import darknet

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def run():
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Deep SORT
    model_filename = 'deep_sort_yolov4/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    file_path = 'omni_people.mp4'
    video_capture = cv2.VideoCapture(file_path)

    if writeVideo_flag:
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('out.mp4', fourcc, video_fps, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
             break

        t1 = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        darknet_detections = darknet.detect_image(network, class_names, darknet_image, thresh=.25)
        darknet.free_image(darknet_image)

        detections_adjusted = []
        for label, confidence, bbox in darknet_detections:
            bbox_adjusted = convert2original(frame, bbox)
            conv_bbox = (bbox_adjusted[0] - bbox_adjusted[2] / 2, bbox_adjusted[1] - bbox_adjusted[3] / 2,
                         bbox_adjusted[2], bbox_adjusted[3])
            detections_adjusted.append((label, confidence, conv_bbox))
        features = encoder(frame, [detection[2] for detection in detections_adjusted])
        detections = [Detection(detection[2], detection[1], "", feature) for detection, feature in
                                zip(detections_adjusted, features)]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        1.5e-3 * frame.shape[0], (0, 255, 0), 1)

        cv2.imshow('', frame)

        if writeVideo_flag: # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        fps = (fps + (1./(time.time()-t1))) / 2
        print("FPS = %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    network, class_names, class_colors = darknet.load_network(
        "./cfg/yolov4-tiny.cfg",
        "./cfg/coco.data",
        "yolov4-tiny.weights",
        batch_size=1
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    run()
