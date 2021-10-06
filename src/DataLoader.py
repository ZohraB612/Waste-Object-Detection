# Import packages
import numpy as np
import cv2
import os


class DataLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imPths, verbose=-1):
        data = []
        lbls = []

        for (i, imPth) in enumerate(imPths):
            im = cv2.imread(imPth)
            lbl = imPth.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for pp in self.preprocessors:
                    im = pp.preprocessors(im)

            data.append(im)
            lbls.append(lbl)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imPths)))

        return np.array(data), np.array(lbls)
