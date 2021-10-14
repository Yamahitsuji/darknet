import numpy as np
from . import linear_assignment
from .detection import Detection
from .track import Track
from typing import List, Optional, Union


def iou(bbox: Union[Detection, Track], candidates: List[Detection]) -> np.ndarray:
    """Computer intersection over union.
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    tlbr = bbox.to_tlbr()
    ious: np.ndarray = np.zeros(len(candidates), dtype=np.float)
    for i, candidate in enumerate(candidates):
        candidate_tlbr = list(candidate.to_tlbr())
        if candidate_tlbr[2] - candidate.img_w > tlbr[0]:
            candidate_tlbr[0] -= candidate.img_w
            candidate_tlbr[2] -= candidate.img_w
        elif candidate_tlbr[0] + candidate.img_w < tlbr[2]:
            candidate_tlbr[0] += candidate.img_w
            candidate_tlbr[2] += candidate.img_w

        tl_x = max(tlbr[0], candidate_tlbr[0])
        tl_y = max(tlbr[1], candidate_tlbr[1])
        br_x = min(tlbr[2], candidate_tlbr[2])
        br_y = min(tlbr[3], candidate_tlbr[3])
        if tl_x >= br_x or tl_y > br_y:
            ious[i] = 0
            continue

        intersection_area = (br_x - tl_x) * (br_y - tl_y)
        union_area = (tlbr[2] - tlbr[0]) * (tlbr[3] - tlbr[1]) \
                     + (candidate_tlbr[2] - candidate_tlbr[0]) * (candidate_tlbr[3] - candidate_tlbr[1]) \
                     - intersection_area
        ious[i] = intersection_area / union_area

    return ious


def iou_cost(tracks: List[Track], detections: List[Detection], track_indices: Optional[List[int]] = None,
             detection_indices: Optional[List[int]] = None) -> np.ndarray:
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[Track]
        A list of tracks.
    detections : List[Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix: np.ndarray = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx]
        candidates: List[Detection] = [detections[i] for i in detection_indices]
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
