from pathlib import Path
from typing import Optional

import PIL.Image
import numpy as np


class Page:
    def __init__(self, path: Optional[Path] = None, pixels: np.array = None):
        if path is None:
            self._key = None
            self._pixels = pixels
        else:
            path = Path(path)

            self._key: str = str(path.absolute())

            if not path.is_file():
                raise FileNotFoundError(path)

            im = PIL.Image.open(str(path.absolute())).convert("RGB")
            self._pixels: np.array = np.array(im)

            # self._pixels = cv2.imread(
            #	str(path.absolute()), cv2.IMREAD_COLOR)

            if self._pixels is None:
                raise ValueError("issue in loading image %s" % path)

    @property
    def key(self) -> str:
        return self._key

    @property
    def pixels(self) -> np.array:
        return self._pixels

    @property
    def shape(self) -> tuple:
        return self._pixels.shape[:2]

    @property
    def size(self) -> tuple:
        return tuple(reversed(self.shape))

    @property
    def extent(self) -> float:
        return np.linalg.norm(np.array(self.size))

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def annotate(self, **kwargs):
        import annotations
        return annotations.Annotate().page(self, **kwargs)
