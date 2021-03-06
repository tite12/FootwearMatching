import numpy as np
import cv2 as cv
import math
import util

def fuzzyEnhancement(img) :
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

    return img

def adaptiveEnhancement(img) :
    #enhancing degree, should be between 25 and 50
    c = 25.0
    #enhancement range, (0,1)
    b = 0.8
    #wavelets
    #TODO: get it work with the wavelets
    # LL, (LH, HL, HH) = pywt.dwt2(img, 'bior1.3')
    # kernel = np.array([[-1, -1, -1],
    #                    [-1, 8, -1],
    #                    [-1, -1, -1]])
    # LL = ndimage.convolve(img, kernel)
    #
    # kernel = np.array([[-1, -1, -1, -1, -1],
    #                    [-1, 1, 2, 1, -1],
    #                    [-1, 2, 4, 2, -1],
    #                    [-1, 1, 2, 1, -1],
    #                    [-1, -1, -1, -1, -1]])
    # LH = ndimage.convolve(img, kernel)
    #
    # HL = ndimage.gaussian_filter(img, 3)
    # HH = img - HL
    # wavelets = [LL, LH, HL, HH]
    # thresholds = []
    # for wavelet in wavelets :
    #     wavelet = np.float32(wavelet)
    #
    #     min, max, minLoc, maxLoc = cv.minMaxLoc(wavelet)
    #     if min < 0 :
    #         wavelet += abs(min)
    #     else :
    #         wavelet -= min
    #     min, max, minLoc, maxLoc = cv.minMaxLoc(wavelet)
    #
    #     wavelet = wavelet / max;
    #
    #     min, max, minLoc, maxLoc = cv.minMaxLoc(wavelet)
    #
    #     height, width = wavelet.shape
    #     mean = cv.mean(wavelet)
    #     coeff = 0;
    #     for x in range(width):
    #         for y in range(height):
    #             coeff += pow(wavelet[y, x] - mean[0], 2)
    #     coeff = coeff / (width * height)
    #     coeff = math.sqrt(coeff) * 0.5
    #     thresholds.append(coeff)
    #
    # cv.imshow("LL", LL)
    # cv.imshow("LH", LH)
    # cv.imshow("HL", HL)
    # cv.imshow("HH", HH)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    thresholds = [0.2, 0.4, 0.6, 0.8]
    thresholds.sort()
    height, width = img.shape
    for x in range(width):
        for y in range(height):
            max = 1
            for th in thresholds :
                if img[y, x] <= th:
                    max = th
                    break
            img[y, x] = a(b, c) * max * (sigma(c * ((img[y, x] / max) - b)) - sigma(-c * ((img[y, x] / max) + b)))

    return img

def sigma(x) :
    res = 1.0 / (1.0 + pow(math.e, -x))
    return res

def a(b, c) :
    res = 1.0 / (sigma(c * (1.0-b))-sigma(-c * (1.0+b)))
    return res

def SMQT(img) :
    outputCode = np.chararray(img.shape, 9)
    outputCode[:] = '0'
    avg = cv.mean(img)
    mask = np.ones(img.shape)
    mask = mask.astype(np.uint8)
    outputCode = SMQTrecursive(img, mask, outputCode, avg, 8)
    height, width = img.shape
    for x in range(width):
        for y in range(height):
            img[y, x] = binaryToDecimal(outputCode[y, x])
    img = img.astype(np.uint8)
    return img

def SMQTrecursive(img, mask, outpyCode, avg, depth) :
    print depth
    if depth <= 0 :
        return outpyCode
    height, width = img.shape
    mask1 = np.zeros(mask.shape)
    mask1 = mask1.astype(np.uint8)
    mask0 = np.zeros(mask.shape)
    mask0 = mask0.astype(np.uint8)
    for x in range(width):
        for y in range(height):

            if mask[y, x] == 1 :
                if img[y, x] <= avg[0] :
                    outpyCode[y, x] = outpyCode[y, x] + '0'
                    mask0[y, x] = 1
                else :
                    outpyCode[y, x] = outpyCode[y, x] + '1'
                    mask1[y, x] = 1

    avg1 = cv.mean(img, mask1)
    avg0 = cv.mean(img, mask0)
    depth -= 1
    outpyCode = SMQTrecursive(img, mask0, outpyCode, avg0, depth)
    outpyCode = SMQTrecursive(img, mask1, outpyCode, avg1, depth)
    return outpyCode

def binaryToDecimal(n):
    return int(n,2)

#img needs to be uint8
def fastSMQT(img) :
    img = util.normalize(img, 255)
    img = img.astype(np.uint8)
    #ordered list of possible values in am image 0-255
    #first row is the frequency of given element
    #second row is the sum of frequencies so far
    #third row is the sum of all elements so far
    occuranceMap = np.zeros((3, 256))
    height, width = img.shape
    #fill the frequencies
    for x in range(width):
        for y in range(height):
            occuranceMap[0, img[y, x]] += 1
    codes = dict()
    #calculate the second and the third row of the occuranceMap
    for x in range(256) :
        if x == 0:
            occuranceMap[1, x] = occuranceMap[0, x]
        else :
            occuranceMap[1, x] = occuranceMap[1, (x - 1)] + occuranceMap[0, x]
            occuranceMap[2, x] = occuranceMap[2, (x - 1)] + (occuranceMap[0, x] * x)
        codes[x] = "0"

    avg = occuranceMap[2, 255] / occuranceMap[1, 255]
    codes = fastSMQTrecursive(occuranceMap, codes, avg, 0, 255, 8)

    for x in range(width):
        for y in range(height):
            img[y, x] = binaryToDecimal(codes[img[y, x]][0:9])

    return img

def fastSMQTrecursive(occuranceMap, codes, avg, valueFrom, valueTo, depth) :
    if depth <= 0 :
        return codes

    biggestSmallerValue = -1;
    #values from valueFrom to valueTo exluding valueTo
    for x in range(valueFrom, valueTo + 1) :
        if x <= avg :
            codes[x] = codes[x] + "0"
            if x > biggestSmallerValue and occuranceMap[0, x] > 0:
                biggestSmallerValue = x
        else :
            codes[x] = codes[x] + "1"

    depth -= 1
    if biggestSmallerValue > -1 :
        avg0 = occuranceMap[2, biggestSmallerValue] / occuranceMap[1, biggestSmallerValue]
        if valueFrom > 0 and valueFrom != biggestSmallerValue:
            avg0 = (occuranceMap[2, biggestSmallerValue] - occuranceMap[2, valueFrom])/ (occuranceMap[1, biggestSmallerValue] - occuranceMap[1, valueFrom])
        codes = fastSMQTrecursive(occuranceMap, codes, avg0, valueFrom, biggestSmallerValue, depth)

    #TODO: there occurs a weird runtime warning
    avg1 = (occuranceMap[2, valueTo] - occuranceMap[2, biggestSmallerValue]) / (occuranceMap[1, valueTo] - occuranceMap[1, biggestSmallerValue])
    codes = fastSMQTrecursive(occuranceMap, codes, avg1, biggestSmallerValue + 1, valueTo, depth)

    return codes