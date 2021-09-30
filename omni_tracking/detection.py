import math
from typing import List, Tuple
import numpy as np


class Detection:
    """
        This class represents a bounding box detection in a single image.

        Parameters
        ----------
        theta: float
            Yaw angle rad of top left. 0 <= theta < 2π.
        phi: float
            ywa angle rad of top left. -π/2 <= phi <= π/2.
        width : int
            Bounding box width.
        height : int
            Bounding box height.
        confidence : float
            Detector confidence score.
        feature : List[float]
            A feature vector that describes the object contained in this image.
    """
    theta: float
    phi: float
    height: int
    width: int
    confidence: float
    feature: np.ndarray
    img_w: int
    img_h: int

    def __init__(self, theta: float, phi: float, width: int, height: int, confidence: float,
                 feature: np.ndarray, img_w: int, img_h: int) -> None:
        self.theta = theta
        self._normalize_theta()
        self.phi = phi
        self.width = width
        self.height = height
        self.confidence = confidence
        self.feature = feature
        self.img_w = img_w
        self.img_h = img_h

    def to_tlwh(self) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to format `(top left x, top left y, width, height)`
        """
        tl_x = int(self.theta / math.tau * self.img_w)
        tl_y = int(self.img_h / 2 + self.phi / (math.pi / 2) * self.img_h / 2)
        return tl_x, tl_y, self.width, self.height

    def to_tlbr(self) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to format `(top left x, top left y, bottom right x, bottom right y)`.
        """
        tlwh = self.to_tlwh()
        return tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]

    def to_scpah(self) -> Tuple[float, float, float, float, int]:
        """
            Convert bounding box to format `(sin θ, cos θ, φ, aspect ratio,
            height)`, where the aspect ratio is `width / height`.
        """
        return math.sin(self.theta), math.cos(self.theta), self.phi, self.width / self.height, self.height

    def _normalize_theta(self) -> None:
        while self.theta < 0 or math.tau <= self.theta:
            if self.theta < 0:
                self.theta = self.theta + math.tau
            elif math.tau <= self.theta:
                self.theta = self.theta - math.tau
