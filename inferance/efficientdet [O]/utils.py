import numpy as np
import math
import cv2

def plot_boxes_cv2(img, det_result, class_names=None):
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    w = img.shape[1]
    h = img.shape[0]

    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    # [classID, score , x, y, x, y]
    for det in det_result:
        c1 = (int(det[2] * w), int(det[3] * h))
        c2 = (int(det[4] * w), int(det[5] * h))

        rgb = (255, 0, 0)

        if class_names:
            cls_conf = det[1]
            cls_id = det[0]
            label = class_names[cls_id]
            print("{}: {}".format(label, cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)

            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cc2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, cc2, rgb, -1)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        img = cv2.rectangle(img, c1, c2, rgb, tl)
    return img

