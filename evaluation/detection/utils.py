import numpy as np
import math
import cv2

def sort_score_index(scores, threshold=0.0, top_k=0, descending=True):
    score_index = []
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
    # print(score_index)
    if not score_index:
        return []

    np_scores = np.array(score_index)
    if descending:
        np_scores = np_scores[np_scores[:, 0].argsort()[::-1]]
    else:
        np_scores = np_scores[np_scores[:, 0].argsort()]

    if top_k > 0:
        np_scores = np_scores[0:top_k]
    return np_scores.tolist()


def nms(boxes, scores, iou_threshold=0.5, score_threshold=0.3, top_k=0):
    if scores is not None:
        scores = sort_score_index(scores, score_threshold, top_k)
        if scores:
            order = np.array(scores, np.int32)[:, 1]
        else:
            return []
    else:
        y2 = boxes[:3]
        order = np.argsort(y2)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    while len(order) > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= iou_threshold)[0]
        # over -> order
        order = order[inds + 1]

    return keep


def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names