import numpy as np
import math
from typing import List, Optional, Tuple
from .detection import Detection
from .kalman_filter import KalmanFilter


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative: int = 1
    Confirmed: int = 2
    Deleted: int = 3


class Track:
    """
    A single target track with state space `(sin_theta, cos_theta, phi, a, h)` and associated
    velocities, where `(sin_theta, cos_theta, phi)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """
    mean: np.ndarray
    covariance: np.ndarray
    img_w: int
    img_h: int
    track_id: int
    hits: int
    age: int
    time_since_update: int
    state: int
    features: List[np.ndarray]
    _n_init: int
    _max_age: int

    def __init__(self, mean: np.ndarray, covariance: np.ndarray, img_w: int, img_h: int, track_id: int, n_init: int,
                 max_age: int, feature: Optional[np.ndarray] = None) -> None:
        self.mean = mean
        self.covariance = covariance
        self.img_w = img_w
        self.img_h = img_h
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self) -> Tuple[int, int, int, int]:
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
            The bounding box.
        """
        h = int(self.mean[4])
        w = int(self.mean[3] * h)
        sin_theta: float = self.mean[0]
        cos_theta: float = self.mean[1]
        theta: float = _normalize_theta(math.atan2(sin_theta, cos_theta))
        phi: float = self.mean[2]
        x = int(theta / math.tau * self.img_w)
        y = int(phi / (math.pi / 2) * self.img_h / 2 + self.img_h / 2)

        return x, y, w, h

    def to_tlbr(self) -> Tuple[int, int, int, int]:
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
            The bounding box.
        """
        tlwh = self.to_tlwh()
        return tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

    def predict(self, kf: KalmanFilter) -> None:
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, detection: Detection) -> None:
        """Perform Kalman filter measurement update step and update the feature
        cache.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, np.array(detection.to_scpah()))
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted


def _normalize_theta(theta: float) -> float:
    while theta < 0 or math.tau <= theta:
        if theta < 0:
            theta = theta + math.tau
        elif math.tau <= theta:
            theta = theta - math.tau
    return theta
