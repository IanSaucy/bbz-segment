import numpy as np

from Polygons.LabeledPage import LabeledPage

labels = np.load('./8k71pf49w_8k71pf51x-labels.sep.npy')

page = LabeledPage(labels, (3672, 5298))
res = page._find_horz_sep_in_range(860, 1540, 240, 5279)
print(res)
#res = page.find_all_vertical_sep()
#print(res)

# Remove everything but vertical separators
# labels_vert = reduce_to_single_label(labels, VERT)
# labels = reduce_to_single_label(labels, VERT)
# resized = cv.resize(labels_vert, (3672, 5298), interpolation=cv.INTER_AREA)
