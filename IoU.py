# Import packages
import cv2
from collections import namedtuple

det = namedtuple("Detection", ["image_path", "gt", "pred"])


def bound_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    i_o_u = intersection_area / float(box1_area + box2_area - intersection_area)
    return i_o_u


# define the list of example detections ######
examples = [
    det("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
    det("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
    det("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
    det("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
    det("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]

for detection in examples:
    im = cv2.imread(detection.image_path)
    cv2.rectangle(im, tuple(detection.gt[:2]),
                  tuple(detection.gt[2:]), (0, 255, 0), 2)
    cv2.rectangle(im, tuple(detection.pred[:2]),
                  tuple(detection.pred[2:]), (0, 0, 255), 2)

    iou = bound_box_iou(detection.gt, detection.pred)
    cv2.putText(im, "IoUL {:.4f}".format(iou), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print("{}: {:.4f".format(detection.image_path, iou))

    cv2.imshow("Image", im)
    cv2.waitKey(0)

# Think about implementing GIoULoss loss metric (Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression)