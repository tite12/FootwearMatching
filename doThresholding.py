import cv2 as cv
import numpy as np
import math

def blur(img) :
    return cv.medianBlur(img, 3);

def adaptiveThreshold(img) :
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

def otsuThreshold(img) :
    retval, img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return img

def fuyyzEnhancement(img) :
    orig = img.copy()
    min, max, minLoc, maxLoc = cv.minMaxLoc(img)
    height, width = img.shape
    for x in range(width):
        for y in range(height):
            currentVal = img[y, x]
            currentVal = (currentVal - min)/(max-min)

            if currentVal < 0.5 :
                currentVal = 2 * pow(currentVal, 2)
            else:
                currentVal = 1 - (2 * pow((1-currentVal), 2))
            img[y, x] = currentVal

    laplaceImg = cv.Laplacian(img, cv.CV_32FC1, 3)

    edgeAverageImg = img.copy()
    for x in range(width):
        for y in range(height):
            #we use 3x3 window, current point is the center
            xCoords = [x]
            yCoords = [y]
            #range have to be odd
            dist = (3 - 1) / 2
            distVal = 1
            while distVal <= dist :
                currVal = x - distVal
                if currVal >= 0  :
                    xCoords.append(currVal)
                currVal = x + distVal
                if currVal < width :
                    xCoords.append(currVal)

                currVal = y - distVal
                if currVal >= 0 :
                    yCoords.append(currVal)
                currVal = y + distVal
                if currVal < height:
                    yCoords.append(currVal)

                distVal += 1

            #window coordinates are given, calculate edge average on that point
            numerator = 0
            denominator = 0
            for windowX in xCoords:
                for windowY in yCoords:
                    #is this ok though?
                    numerator += abs(img[windowY, windowX]) * abs(laplaceImg[windowY, windowX])
                    denominator += abs(laplaceImg[windowY, windowX])
            if denominator == 0 :
                edgeAverageImg[y, x] = 0
            else :
                edgeAverageImg[y, x] = (numerator / denominator)

    contrastImage = img.copy()
    for x in range(width):
        for y in range(height):
            numerator = abs(img[y, x] - edgeAverageImg[y, x])
            denominator = abs(img[y, x] + edgeAverageImg[y, x])
            if denominator == 0 :
                contrastImage[y, x] = 0
            else :
                contrastImage[y, x] = numerator / denominator

    amplificationImg = img.copy()
    minContrast, maxContrast, minLoc, maxLoc = cv.minMaxLoc(contrastImage)
    alpha = 0.3
    for x in range(width):
        for y in range(height):
            amplificationFactor = 1.0 - alpha * math.cos((math.pi / 2)*((contrastImage[y, x] - minContrast)/(maxContrast - minContrast)))
            contrastImage[y, x] = pow(contrastImage[y, x], amplificationFactor)

    for x in range(width):
        for y in range(height):
            #TODO: im (not)using absolute values here, is it ok?
            if img[y, x] <= edgeAverageImg[y, x] :
                img[y, x] = edgeAverageImg[y, x] * ((1-contrastImage[y, x])/(1+contrastImage[y, x]))
            else :
                img[y, x] = edgeAverageImg[y, x] * ((1 + contrastImage[y, x]) / (1 - contrastImage[y, x]))

            img[y, x] = img[y, x] * (max - min) + min


    cv.imshow("img", img)
    cv.imshow("contrast", contrastImage)
    cv.imshow("edgeAverage", edgeAverageImg)
    cv.imshow("amplification", amplificationImg)
    cv.imshow("orig", orig)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img