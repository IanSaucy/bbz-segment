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


def colorize(labels: np.array) -> PIL.Image:
    """"
    Build a color image based on the labels of this image. Essentially converts the labels
    such as 0, 1, 2, or 3(representing different types of separators of this image) to RGB colors
    for easier output and readability
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
