import numpy as np

def center_to_corner(boxes):
    #Converting bounding boxes from center format(cx, xy, width, weight) to corner format(xmin, ymin, xmax, ymax)
    temp = boxes.copy()
    temp[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2)  # xmin
    temp[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2)  # ymin
    temp[..., 2] = boxes[..., 0] + (boxes[..., 2] / 2)  # xmax
    temp[..., 3] = boxes[..., 1] + (boxes[..., 3] / 2)  # ymax
    return temp