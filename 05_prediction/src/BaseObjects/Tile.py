import itertools
import math

import numpy as np


class Tile:
    def __init__(self, outer, inner):
        self._outer = outer
        self._inner = inner

    @property
    def outer(self):
        return self._outer

    @property
    def inner(self):
        return self._inner

    def read_outer(self, pixels):
        x0, y0, x1, y1 = self._outer
        return pixels[y0:y1, x0:x1]

    def write_inner(self, labels, data):
        """
        Fills the labels input array with the data specified
        but along the region specified by the tile
        Args:
            labels (): An NP array larger than data
            data (): The data to be aligned to the region of the labels input

        Returns: Void
        """
        x0, y0, x1, y1 = self._inner
        dx, dy = np.array(self._inner[:2]) - np.array(self._outer[:2])
        labels[y0:y1, x0:x1] = data[dy:dy + (y1 - y0), dx:dx + (x1 - x0)]


class Tiles:
    """
    A class to handle tiling of images. That is, breaking them down into smaller sub-images
    """

    def __init__(self, tile_size, beta0=50):
        self._tile_size = tile_size
        assert all(beta0 < s for s in tile_size)
        self._beta0 = beta0

    def _tiles_1(self, full_size, tile_size):
        if tile_size == full_size:
            yield (0, full_size), (0, full_size)
        else:
            n_steps = math.ceil(full_size / tile_size)

            # TODO Figure out what this does
            while True:
                r = (full_size - tile_size) / ((n_steps - 1) * tile_size)

                if tile_size * (1 - r) > self._beta0:
                    break

                n_steps += 1

            x0 = []
            x1 = []
            for i in range(n_steps):
                x = round(i * tile_size * r)
                x -= max(0, x + tile_size - full_size)

                x0.append(x)
                x1.append(x + tile_size)

            for i in range(n_steps):
                if i > 0:
                    x0_inner = (x1[i - 1] + x0[i]) // 2
                else:
                    x0_inner = 0

                if i < n_steps - 1:
                    x1_inner = (x1[i] + x0[i + 1]) // 2
                else:
                    x1_inner = full_size

                yield (x0[i], x1[i]), (x0_inner, x1_inner)

    def __call__(self, full_size):
        p = [list(self._tiles_1(f, t)) for f, t in zip(full_size, self._tile_size)]
        for x, y in itertools.product(*p):
            (x0, x1), (xi0, xi1) = x
            (y0, y1), (yi0, yi1) = y
            yield Tile((x0, y0, x1, y1), (xi0, yi0, xi1, yi1))
