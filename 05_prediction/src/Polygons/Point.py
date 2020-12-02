from collections import namedtuple
from enum import Enum, auto
from functools import total_ordering
from typing import Optional, NamedTuple, List


class SeparatorTypes(Enum):
    Vertical = auto()
    Horizontal = auto()


@total_ordering
class Point(NamedTuple):
    row: int
    col: int

    def __str__(self) -> str:
        return f'(row={self.row}, col={self.col})'

    def __lt__(self, other):
        return self.col < other.col


@total_ordering
class VerticalSeparator:

    def __init__(self, top: Point, bottom: Point):
        self._top_point: Point = top
        self._bottom_point: Point = bottom

    def __str__(self) -> str:
        return f'Top: {self._top_point}, Bot: {self._bottom_point}'

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: object) -> bool:
        if isinstance(other, VerticalSeparator):
            return self._top_point < other._top_point
        else:
            return NotImplemented

    @property
    def top_point(self):
        return self._top_point

    @property
    def bottom_point(self):
        return self._bottom_point

    @top_point.setter
    def top_point(self, value):
        self._top_point = value

    @bottom_point.setter
    def bottom_point(self, value):
        self._bottom_point = value


class Box(NamedTuple):
    top_left: Point
    top_right: Point
    bot_left: Point
    bot_right: Point


class Article:
    def __init__(self, boxes: List[Box]):
        self._boxes = boxes

    def __str__(self):
        return f'{self.get_boxes()}'

    def add_box(self, box: Box):
        self._boxes.append(box)

    def get_boxes(self):
        return self._boxes


class HorizontalSeparator:
    def __init__(self, left: Point, right: Point):
        self.left_point = left
        self.right_point = right

    def __str__(self):
        return f'Left: {self.left_point} Right: {self.right_point}'

    def __repr__(self):
        return self.__str__()

    @property
    def left_point(self):
        return self._left_point

    @property
    def right_point(self):
        return self._right_point

    @left_point.setter
    def left_point(self, value):
        self._left_point = value

    @right_point.setter
    def right_point(self, value):
        self._right_point = value
