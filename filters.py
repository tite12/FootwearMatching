import numpy as np
import cv2 as cv
import math
import util
from matplotlib import pyplot as plt
from skimage import restoration
from scipy.signal import convolve2d

import histogramOperations
import doThresholding

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
    res = np.zeros(img.shape, np.uint8)
    mask = np.zeros(img.shape, np.uint8)
    min, max, minLoc, maxLoc = cv.minMaxLoc(diffImg)
    thresholds = []
    masks = []
    mapping = np.zeros(img.shape, np.uint8)
    classes = 45
    for i in range(1, classes + 1) :
        thresholds.append(calcTh(i, min, max, classes))
        masks.append(np.zeros(img.shape, np.uint8))
    grayVal = 255/classes
    for x in range(width):
        for y in range(height):
            currentVal = diffImg[y, x]
            index = 0
            for th in thresholds :

                if index == 0 and currentVal < th:
                    setPixel(masks[index], x, y, 255)
                    mapping[y, x] = index
                    break
                # res[y, x] = grayVal * index
                # setPixel(res, x, y, grayVal * index, 9)
                setPixel(mask, x, y, 255)

                if index > 0 and currentVal > thresholds[index-1] and currentVal <= th  :
                    setPixel(masks[index], x, y, 255)
                    mapping[y, x] = index
                    break
                index += 1


    means = []
    for maskImg in masks :
        mean = cv.mean(img, maskImg)
        means.append(mean[0])
    for x in range(width):
        for y in range(height):
            res[y,x] = means[mapping[y, x]]
    resLocal = np.zeros(res.shape, np.uint8)
    n = 30
    for x in range(width):
        for y in range(height):
            currentMask = np.zeros(res.shape, np.uint8)
            roiX = 0
            roiY = 0
            roiWidth =0
            roiHeight = 0
            if x < n :
                roiX = 0
                roiWidth = n + x
            else :
                roiX = x - n
                if x + n > width :
                    roiWidth = n + (width - x)
                else :
                    roiWidth = 2 * n

            if y < n :
                roiY = 0
                roiHeight = y + n
            else :
                roiY = y - n
                if y + n > height :
                    roiHeight = n + (height - y)
                else :
                    roiHeight = 2 * n
            truncated = masks[mapping[y, x]][roiY:roiY+roiHeight, roiX:roiX+roiWidth]
            currentMask[roiY:roiY+roiHeight, roiX:roiX+roiWidth] = truncated
            # cv.imshow("truncated", truncated)
            # cv.imshow("current Mask", currentMask)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            mean = cv.mean(img, currentMask)
            resLocal[y, x] = mean[0]





    cv.imshow("mask", mask)
    cv.imshow("region", resLocal)
    cv.imshow("non-local mean", res)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return res

def calcTh(var, lMin, lMax, n) :
    return lMin + (var * (lMax-lMin)/n)

def setPixel(img, x, y, val, nb = 5) :
    img[y, x] = val
    height, width = img.shape
    if x > 0 :
        img[y, x-1] = val
        if nb == 9 :
            if y > 0 :
                img[y-1, x - 1] = val
            if y < height - 1:
                img[y + 1, x-1] = val

    if x < width - 1 :
        img[y, x+1] = val
        if nb == 9:
            if y > 0 :
                img[y-1, x + 1] = val
            if y < height - 1:
                img[y + 1, x+1] = val

    if y > 0 :
        img[y - 1, x] = val
    if y < height - 1 :
        img[y+1, x] = val
    return img