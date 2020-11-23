from typing import List

import numpy as np
import typing

from Predictors import NetPredictor
from Predictors.Predictor import Predictor
from Predictors.Prediction import Prediction


class VotingPredictor(Predictor):
    def __init__(self, *predictors: NetPredictor):
        # Verifies that the passed list of predictors all have the same type
        if not all(p.type == predictors[0].type for p in predictors):
            raise RuntimeError("predictor need to have same pr types")
        self._predictors: NetPredictor = predictors
        self._undecided = predictors[0].background.value

    def __call__(self, pixels) -> Prediction:
        # Obtain raw predictions from each model(the page is run through
        # multiple models)
        predictions = [p(pixels) for p in self._predictors]
        return Prediction(
            predictions[0].page,
            # Build common consensus among all predictions from model
            _majority_vote([p.labels for p in predictions], self._undecided),
            # All possible classes of labels
            self._predictors[0].classes)


def _majority_vote(data: List[np.ndarray], undecided=0) -> np.ndarray:
    """
    A function to find the most common prediction among the list of input labels.
    TODO: Verify that this is indeed what this func does
    Args:
        data (): A list of predicted labels from a page
        undecided (): Value to use in the case there is a tie

    Returns:
        numpy array that is the "common consensus" among the input
    """
    data = np.array(data, dtype=data[0].dtype)
    n_labels = np.max(data) + 1

    counts = np.zeros(
        (n_labels,) + data[0].shape, dtype=np.int32)
    # Build up a count of each label in the list of
    # input labels
    for label in range(n_labels):
        for pr in data:
            counts[label][pr == label] += 1
    # Stack all counts into a depth wise sequence
    # https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
    counts = np.dstack(counts)

    order = np.argsort(counts)
    candidates_count = np.take_along_axis(counts, order[:, :, -2:], axis=-1)
    tie = np.logical_not(candidates_count[:, :, 0] < candidates_count[:, :, 1])

    most_freq = np.argmax(counts, axis=-1).astype(data.dtype)
    most_freq[tie] = undecided

    return most_freq
