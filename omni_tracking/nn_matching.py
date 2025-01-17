from typing import Optional, Callable, Dict, List
import numpy as np


def _cosine_distance(a: np.ndarray, b: np.ndarray, data_is_normalized: Optional[bool] = False) -> np.ndarray:
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : ndarray
        An NxM matrix of N samples of dimensionality M.
    b : ndarray
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric:
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """
    _metric: Callable[[np.ndarray, np.ndarray], np.ndarray]
    matching_threshold: float
    samples: Dict[int, List[np.ndarray]]

    def __init__(self, matching_threshold: float):
        self._metric = _nn_cosine_distance
        self.matching_threshold = matching_threshold
        self.samples = {}

    def partial_fit(self, features: np.ndarray, targets: np.ndarray, active_targets: List[int]) -> None:
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(np.array(self.samples[target]), features)
        return cost_matrix
