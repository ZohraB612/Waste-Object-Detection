# Import packages
import os
import cv2
import config
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

print("[INFO] loading dataset")
data = []
lbls = []
bound_boxes = []
im_paths = []

for csvPath in paths.list_files(config.annots_path, validExts=".csv"):
    r = open(csvPath).read().strip().split("\n")

    for row in r:
        row = row.split(",")
        (filename, shape, startX, startY, endX, endY, label) = row

        imagePath = os.path.sep.join([config.im_path, label, filename])
        im = cv2.imread(imagePath)
        (h, w) = im.shape[:2]

        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        im = load_img(imagePath, target_size=(224, 224))
        im = img_to_array(im)

        data.append(im)
        lbls.append(label)
        bound_boxes.append((startX, startY, endX, endY))
        im_paths.append(imagePath)

data = np.array(data, dtype="float32") / 255.0
lbls = np.array(lbls)
bound_boxes = np.array(bound_boxes, dtype="float32")
im_paths = np.array(im_paths)

lb = LabelBinarizer()
lbls = lb.fit_transform(lbls)

if len(lb.classes_) == 1:
    lbls = to_categorical(lbls)

split = train_test_split(data, lbls, bound_boxes, im_paths, test_size=0.20, random_state=42)

(train_im, test_im) = split[:2]
(train_lbls, test_lbls) = split[2:4]
(train_bound_boxes, test_bound_boxes) = split[4:6]
(train_paths, test_paths) = split[6:]

print("[INFO] saving testing image paths...")
f = open(config.test_paths, "w")
f.write("\n".join(test_paths))
f.close()
