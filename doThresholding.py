import cv2 as cv
import numpy as np
import math
from scipy import ndimage
import pywt
import util

def adaptiveThreshold(img) :
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

def otsuThreshold(img) :
    retval, img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return img

def niblackThreshold(img, binMethod = 0, blockSize = 5, k = 0.5) :
    binarization = cv.ximgproc.BINARIZATION_NIBLACK
    if binMethod == 1 :
        binarization = cv.ximgproc.BINARIZATION_SAUVOLA
    elif binMethod == 2 :
        binarization = cv.ximgproc.BINARIZATION_WOLF
    elif binMethod == 3 :
        binarization = cv.ximgproc.BINARIZATION_NICK
    dst = np.zeros(img.shape)
    # dst = cv.ximgproc.niBlackThreshold(img, 1.0, cv.THRESH_BINARY, blockSize, k)
    dst = cv.ximgproc.niBlackThreshold(img, 1.0, cv.THRESH_BINARY, blockSize, k, dst, binarization)
    return dst

def calculateSkeleton(img) :
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv.threshold(img, 127, 255, 0)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def thinning(img) :
    cv.ximgproc.thinning(img, img)
    return img