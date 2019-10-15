import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import restoration
from scipy.signal import convolve2d

def watershed(img) :

    #otsu thresholdong
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    out = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(out, markers)
    out[markers == -1] = [255, 0, 0]

    return out

def median(img, kernelSize) :
    img = cv.medianBlur(img, kernelSize)
    return img;

def bilateralFilter(img, kernelSize) :
    img = cv.bilateralFilter(img, kernelSize, kernelSize * 2, kernelSize / 2)
    return img

def wiener(img) :
    psf = np.ones((5, 5)) / 25
    img = convolve2d(img, psf, 'same')
    img += 0.1 * img.std() * np.random.standard_normal(img.shape)
    deconvolved_img = restoration.wiener(img, psf, 1100)
    # deconvolved_img = restoration.unsupervised_wiener(img, psf)
    return deconvolved_img

def pde(img, iteration = 5, step = 1.0, edgeSensitivity = 0.02):
    dst = cv.ximgproc.anisotropicDiffusion(img, step, edgeSensitivity, iteration)
    return dst

def epf(img, kernel = 9, th = 20) :
    dst = 	cv.ximgproc.edgePreservingFilter(img, kernel, th)
    return dst