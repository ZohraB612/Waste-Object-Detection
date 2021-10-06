# Import packages
import cv2
import pickle
import config
import imutils
import argparse
import mimetypes
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Argparse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input image/text file of image paths")
args = vars(ap.parse_args())

# Input file
filetype = mimetypes.guess_type(args["input"])[0]
im_paths = [args["input"]]

if "text/plain" == filetype:
    im_paths = open(args["input"]).read().strip().split("\n")

print("[INFO] loading object detector...")
model = load_model(config.model_path)
lb = pickle.loads(open(config.lb_path, "rb").read())

for im_path in im_paths:
    im = load_img(im_path, target_size=(224, 224))
    im = img_to_array(im) / 255.0
    im = np.expand_dims(im, axis=0)

    (box_preds, lbl_preds) = model.predict(im)
    (startX, startY, endX, endY) = box_preds[0]

    i = np.argmax(lbl_preds, axis=1)
    lbl = lb.classes_[i][0]

    im = cv2.imread(im_path)
    im = imutils.resize(im, width=600)
    (h, w) = im.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(im, lbl, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Output", im)
    cv2.waitKey(0)
