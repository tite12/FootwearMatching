import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def calculateHistogram(img) :
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    # width, height = img.shape
    # hist = hist / (width * height)
    plt.hist(img.ravel(), 256, [0, 256]);
    plt.show()
    return

def equalizeHistogram(img) :
    cv.equalizeHist(img, img)
    return img