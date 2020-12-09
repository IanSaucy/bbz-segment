from typing import Tuple

import PIL.Image
import numpy as np
from matplotlib import pyplot as plt


class Prediction:
    def __init__(self, page, labels, classes):
        self._page = page
        self._labels = labels
        self._classes = classes
        self._blocks = None
        self._separators = None
        self._background = self._classes["BACKGROUND"]

    @property
    def background_label(self):
        return self._background

    @property
    def page(self):
        return self._page

    @property
    def labels(self):
        return self._labels

    @property
    def labels_np(self):
        "Return labels as a numpy array"
        return np.asarray(self._labels)

    @property
    def classes(self):
        return self._classes

    def save(self, path):
        """
        Colorize and save the image stored on this object
        to the provided path
        Args:
            path (): The path of where to sace this image
        Returns:
            Void

        """
        colorize(self._labels).save(path)

    def save_labels_array(self, path: str, original_dimensions: Tuple[int, int], input_filename: str) -> None:
        """
        Saves the label 2D array to a numpy file along with image original dimensions
        Args:
            path (): Path and file name of where to save labels
            original_dimensions: Original dimensions of labeled image being saved. In
                (height, width) tuple form.
            input_filename: The filename of the photo that was used to generate this labeling
        Returns:
            None
        """
        np.save(path, {"labels": self.labels_np, "dimensions": original_dimensions, "filename": input_filename})


def colorize(labels: np.array) -> PIL.Image:
    """"
    Build a color image based on the labels of this image.
    Args:
        labels (): The labels to be written over the original image
    Returns:
        PIL.Image with colored regions derived from the labels
    """
    # n_labels = np.max(labels) + 1 # Not used anymore
    colors = category_colors()

    im = PIL.Image.fromarray(labels, "P")
    pil_pal = np.zeros((768,), dtype=np.uint8)
    pil_pal[:len(colors)] = colors
    im.putpalette(pil_pal)

    return im


def category_colors() -> np.array:
    colors = plt.get_cmap("tab10").colors
    return np.array(list(colors)).flatten() * 255
