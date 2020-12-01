import operator
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np

from Polygons.Constants import ExpectedValues
from Polygons.Labels import Labels
from Polygons.Point import Point, VerticalSeparator, HorizontalSeparator
import cv2 as cv


class LabeledPage:
    """
    A class to handle labeled pages and extracting their features
    """

    """
    The buffer we define that must exist around any given vertical separator.
    This helps to make sure we don't detect the same vertical separator multiple times
    while searching the top and bottom of the image. This value is HIGHLY dependent on the input 
    image sizes and resolutions. It also has a large affect on the performance of the search function
    """
    _separator_buffer_range: int = 150

    _horizontal_expected_width: int = 35

    def __init__(self, img: np.array, original_size: Tuple[int, int]):
        self.img = img
        # Create single labeled copies of input image
        self._only_vertical_labels = cv.resize(self._reduce_to_single_label(Labels.Vert), original_size,
                                               interpolation=cv.INTER_AREA)
        self._only_horizontal_labels = cv.resize(self._reduce_to_single_label(Labels.Horz), original_size,
                                                 interpolation=cv.INTER_AREA)

    def _find_horz_sep_in_range(self, start_col: int, end_col: int, start_row: int, end_row: int) -> List[
        Optional[HorizontalSeparator]]:
        if self._only_vertical_labels.shape[1] < end_col:
            raise IndexError
        if start_col < 0:
            raise IndexError
        img = self.only_horizontal_labels
        horz_sep_list: List[HorizontalSeparator] = []
        label_history: List[int] = []
        for row in range(start_row, end_row):
            if Labels.Horz in img[row, start_col:end_col] and Labels.Horz not in label_history:
                horz_sep_list.append(HorizontalSeparator(Point(row, start_col), Point(row, end_col)))
                label_history.append(Labels.Horz)
            else:
                label_history.append(-1)
            # This makes sure we don't input the same separator multiple times since it has a width greater
            # a single pixel
            if len(label_history) > self._horizontal_expected_width:
                label_history.pop(0)
        return horz_sep_list

    def find_all_vertical_sep(self) -> List[VerticalSeparator]:
        top_of_separators: List[Point] = self._find_top_of_vert_sep()
        bot_of_separators: List[Point] = self._find_bot_of_vert_sep()
        # Make sure both have the same number of points
        assert len(top_of_separators) == len(bot_of_separators)
        # Sort to make it easier to match up the bottom and top points of a given separator
        top_of_separators.sort()
        bot_of_separators.sort()
        complete_vertical_separator: List[VerticalSeparator] = []
        for top_point in top_of_separators:
            for bot_point in bot_of_separators:
                # TODO: It would be much better to remove elems from the array as we go.
                # But given that we know that they're going to be small, less than whatever our max number of columns is
                # we can ignore this and just do an O(n^2) search.
                if bot_point.col - self._separator_buffer_range < top_point.col < bot_point.col + self._separator_buffer_range:
                    # These are a match
                    complete_vertical_separator.append(VerticalSeparator(top_point, bot_point))
        assert len(complete_vertical_separator) == len(bot_of_separators)
        # Now we add "soft" left and right separators that are just the edge of the images
        
        complete_vertical_separator.sort()
        return complete_vertical_separator

    def _find_bot_of_vert_sep(self) -> List[Point]:
        img = self.only_vertical_labels
        # height, width = np.shape
        top_search_limit: int = img.shape[0] // 3
        expected_number_seps: int = self._find_number_vert_sep(self._only_vertical_labels)
        number_found_seps: int = 0
        found_separator_ranges = set()
        found_separators: List[Point] = []
        for row in reversed(range(top_search_limit, img.shape[0])):
            if number_found_seps == expected_number_seps:
                break
            for col in range(img.shape[1]):
                if col in found_separator_ranges:
                    # This is a separator we've already seen
                    continue
                if img[row, col] == Labels.Vert:
                    # This is a separator!
                    found_separators.append(Point(row, col))
                    number_found_seps += 1
                    # Add range to what we've already seen so we won't accidentally double add it to our list
                    found_separator_ranges.update(
                        [x for x in range(col - self._separator_buffer_range, col + self._separator_buffer_range)])
                    break
        return found_separators

    def _find_top_of_vert_sep(self) -> List[Point]:
        img = self.only_vertical_labels
        bottom_search_limit: int = img.shape[0] // 2
        expected_number_seps: int = self._find_number_vert_sep(self._only_vertical_labels)
        number_found_seps: int = 0
        found_separator_ranges = set()
        found_separators: List[Point] = []
        for row in range(bottom_search_limit):
            if number_found_seps == expected_number_seps:
                break
            for col in range(img.shape[1]):
                if col in found_separator_ranges:
                    # This is a separator we've already seen
                    continue
                if img[row, col] == Labels.Vert:
                    # This is a separator!
                    found_separators.append(Point(row, col))
                    number_found_seps += 1
                    # Add range to what we've already seen so we won't accidentally double add it to our list
                    found_separator_ranges.update(
                        [x for x in range(col - self._separator_buffer_range, col + self._separator_buffer_range)])
                    break
        return found_separators

    def _reduce_to_single_label(self, desired_label: Labels) -> np.array:
        """
        Takes the input image and removes all labels except the specified one.
        ! Warning: Always replaces labels to be removed with 0(and there is a label which itself is 0)
        Args:
            desired_label (): Label desired to be kept

        Returns: The modified numpy array

        """
        unique = np.unique(self.img)
        temp_img = np.copy(self.img)
        for label in unique:
            if label != desired_label:
                temp_img[temp_img == label] = 0
        return temp_img

    def _find_number_vert_sep(self, input_image: np.array) -> int:
        """
        Takes the image and finds the number of vertical separators. Searches multiple
        places in the image to ideally come up with the "true" number of vertical separators.
        The number found is checked against a sane max number expected value as a rudimentary
        sanity check.
        Args:
            (x,y) coordinate should either be a vertical separator or the integer value `0`.
            In other words, it expects that the input image is binary, only containing two possible
            values at an (x,y) coordinate.

        Returns: The number of vertical separators found

        """
        start_loc = input_image.shape[0] // 2
        step_size = int(input_image.shape[0] * 0.2)
        locations_to_test = [start_loc - step_size, start_loc, start_loc + step_size]

        vertical_sep_locs = []
        vertical_sep_counts = defaultdict(int)
        sliding_window_history = []
        for row in locations_to_test:
            for col, value in enumerate(input_image[row]):
                if value == Labels.Vert and Labels.Vert not in sliding_window_history:
                    vertical_sep_locs.append(Point(row, col))
                    vertical_sep_counts[row] += 1
                sliding_window_history.append(value)
                if len(sliding_window_history) > 3:
                    sliding_window_history.pop(0)

        # Find the row that has the most number of vertical separators
        max_vert_sep_row = max(vertical_sep_counts.items(), key=operator.itemgetter(1))[0]
        assert vertical_sep_counts[max_vert_sep_row] < ExpectedValues.maxNumberOfVerticalSeparators
        return vertical_sep_counts[max_vert_sep_row]

    @property
    def only_vertical_labels(self):
        return self._only_vertical_labels

    @property
    def only_horizontal_labels(self):
        return self._only_horizontal_labels
