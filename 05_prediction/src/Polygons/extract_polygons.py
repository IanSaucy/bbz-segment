import operator
from collections import namedtuple, Counter, defaultdict
from functools import total_ordering
from typing import List, Tuple

import numpy as np
import cv2 as cv


@total_ordering
class PointOrdering(object):
    __slots__ = ()

    def __lt__(self, other):
        return self.col < other.col


class Point(PointOrdering, namedtuple('Point', 'col row')):
    pass


Separator = namedtuple('Separator', 'top bottom')

TABLE = 0
HORZ = 1
VERT = 2
BACKGROUND = 3
# 0 = Table
# 1 = Horizontal
# 2 = Vertical
# 3 = Background
labels = np.load('./8k71pf49w_8k71pf51x-labels.sep.npy')
resized = cv.resize(labels, (3672, 5298), interpolation=cv.INTER_AREA)

# Count vertical separators in multiple places
# Take middle of image, run test on some percent of pixels around that
start_loc = resized.shape[0] // 2
step_size = int(resized.shape[0] * 0.2)
locations_to_test = [start_loc - step_size, start_loc, start_loc + step_size]
# Verify all are within the bounds of the image
for i, col in enumerate(locations_to_test):
    while HORZ in resized[col] or TABLE in resized[col]:
        # This col contains a horizontal or vertical separator
        # change it slightly so that we don't run into these values
        locations_to_test[i] += 10
    assert 0 < col < resized.shape[0]

# Sliding window of history of previous values in the col.
vertical_sep_locs = []
vertical_sep_counts = defaultdict(int)
sliding_window_history = []
for row in locations_to_test:
    for col, value in enumerate(resized[row]):
        if value == VERT and VERT not in sliding_window_history:
            vertical_sep_locs.append(Point(col, row))
            vertical_sep_counts[row] += 1
        sliding_window_history.append(value)
        if len(sliding_window_history) > 3:
            sliding_window_history.pop(0)


def get_sep_loc(col: int, row: int, input: np.array, search_range: int, search_elem: int) -> Tuple[bool, int, int]:
    leftmost_vert_sep: int = -1
    found_sep = False
    for currCol in range(col - search_range, search_range + col + 1):
        if input[currCol, row] == search_elem:
            leftmost_vert_sep = currCol
            found_sep = True
            break
    return found_sep, leftmost_vert_sep, row


# Get col that has found the most number of separators
max_vert_sep_row = max(vertical_sep_counts.items(), key=operator.itemgetter(1))[0]
# "Sane" max number of columns
# TODO Extract this field into something more pythonic
assert vertical_sep_counts[max_vert_sep_row] < 8
# Filter separator locations down to just the identified values
vertical_sep_locs: List[Point] = list(filter(lambda val: val.row == max_vert_sep_row, vertical_sep_locs))
# Walk up and down separator to find top and bottom
complete_separators = []
for loc in vertical_sep_locs:
    curr_soft_top_row = -1
    curr_soft_top_col = loc.col
    curr_soft_bot_row = -1
    curr_soft_bot_col = loc.col
    for temp_row in reversed(range(0, loc.row + 1)):
        found, new_col, new_row = get_sep_loc(curr_soft_top_col, temp_row, resized, 10, VERT)
        if found:
            curr_soft_top_row = new_row
            curr_soft_top_col = new_col
    print(curr_soft_top_row, curr_soft_top_row)
    # for i in range(loc.y, resized.shape[1]):
    #     if VERT in resized[loc.x][i - 3:i + 3]:
    #         curr_soft_bot = i
    #     else:
    #         break
    # complete_separators.append(Separator(curr_soft_top, curr_soft_bot))

print(max_vert_sep_row)
print(vertical_sep_locs)
print(vertical_sep_counts)
print(complete_separators)
