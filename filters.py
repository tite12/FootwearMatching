import numpy as np
import cv2 as cv
import math
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

def nonLocalMeans(img) :
    dst = cv.fastNlMeansDenoising(img)
    return dst

def regionBasedNonLocalMeans(img) :
    gradX = cv.Sobel(img, cv.CV_16S, 1, 0)
    gradY = cv.Sobel(img, cv.CV_16S, 0, 1)
    cv.convertScaleAbs(gradX, gradX)
    cv.convertScaleAbs(gradY, gradY)

    #TODO: gaussian kernel
    diffImg = np.zeros(img.shape)
    height, width = img.shape
    for x in range(width):
        for y in range(height):
            t11 = pow(gradX[y, x], 2)
            t11 = t11.astype(np.int64)
            t12 = gradX[y, x] * gradY[y, x]
            t12 = t12.astype(np.int64)
            t22 = pow(gradY[y, x], 2)
            t22 = t22.astype(np.int64)
            diff = abs(t11 - t22)
            eigVal1 = 0.5 * (t11 + t22 + math.sqrt(pow(diff, 2) + (4 * pow(t12, 2))))
            eigVal2 = 0.5 * (t11 + t22 - math.sqrt(pow(diff, 2) + (4 * pow(t12, 2))))
            diffImg[y, x] = abs(eigVal1 - eigVal2)
    res = np.zeros(img.shape)
    min, max, minLoc, maxLoc = cv.minMaxLoc(diffImg)
    thresholds = [calcTh(1, min, max, 4), calcTh(2, min, max, 4), calcTh(3, min, max, 4), calcTh(4, min, max, 4)]
    for x in range(width):
        for y in range(height):
            currentVal = diffImg[y, x]
            index = 0
            for th in thresholds :
                if index == 0 and currentVal < th :
                    break
                elif currentVal > thresholds[index-1] and currentVal < th  :
                    print(85*index)
                    res[y, x] = 85*index
                    break
                elif index == 3 :
                    print(85 * index)
                    res[y, x] = 85 * index
                index += 1

    min, max, minLoc, maxLoc = cv.minMaxLoc(res)
    cv.imshow("x", res)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def calcTh(var, lMin, lMax, n) :
    return lMin + (var * (lMax-lMin)/n)
