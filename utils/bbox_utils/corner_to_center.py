import numpy as np

def corner_to_center(boxes):
    #Converting bounding boxes from center format(xmin, ymin, xmax, ymax) to corner format(cx,cy, width, height)
    temp = boxes.copy()
    width = np.abs(boxes[..., 0] - boxes[..., 2])
    height = np.abs(boxes[..., 1] - boxes[..., 3])
    temp[..., 0] = boxes[..., 0] + (width / 2)  # cx
    temp[..., 1] = boxes[..., 1] + (height / 2)  # cy
    temp[..., 2] = width  # xmax
    temp[..., 3] = height  # ymax
    return temp