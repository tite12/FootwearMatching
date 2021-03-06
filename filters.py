import numpy as np
import cv2 as cv
import math
import util
from matplotlib import pyplot as plt
from skimage import restoration
from scipy.signal import convolve2d
from numpy import linalg as la
from scipy.spatial import distance

import operator
import histogramOperations
import doThresholding
import enhancement

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

def nonLocalGrouping(img) :
    blur = cv.GaussianBlur(img, (11, 11), 0)
    gradX = cv.Sobel(blur, cv.CV_16S, 1, 0)
    gradY = cv.Sobel(blur, cv.CV_16S, 0, 1)
    cv.convertScaleAbs(gradX, gradX)
    cv.convertScaleAbs(gradY, gradY)

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
    return diffImg

def regionBasedNonLocalMeans(img, tryEqualize = True, mod = False) :
    diffImg = nonLocalGrouping(img)
    # cv.imshow("diff", util.normalize(diffImg, 1))
    # cv.waitKey(0)
    if mod :
        tmp = diffImg
        tmp[tmp == 0] = np.nan
        med = np.nanmean(tmp)
        tmp[tmp > med] = np.nan
        med = np.nanmean(tmp)
        diffImg[diffImg < med] = 0
        # mean = np.nanmean(diffImg)
        # diffImg[diffImg < mean] = 0
        # diffImg[diffImg == np.nan] = 0
        cv.imshow("after", histogramOperations.equalizeHistogram(np.uint8(util.normalize(diffImg, 255))))
        cv.waitKey(0)

    res = np.zeros(diffImg.shape, np.uint8)

    classes = 25
    maskSize, masks, mapping = fillClasses(diffImg, classes)

    sortedMaskSize = sorted(maskSize.items(), key=operator.itemgetter(1))

    height, width = img.shape
    dataPixels = height * width
    dataPixels = dataPixels * 0.6

    # print("recalculate?")
    # print(int(sortedMaskSize[-1][1]))
    # print dataPixels
    equalized = False
    if tryEqualize :
        if sortedMaskSize[-1][1] > dataPixels :
            diffImg = np.uint8(util.normalize(diffImg, 255))
            diffImg = histogramOperations.equalizeHistogram(diffImg)
            maskSize, masks, mapping = fillClasses(diffImg, classes)
            sortedMaskSize = sorted(maskSize.items(), key=operator.itemgetter(1))
            print("I was here")
            equalized = True

    means = []
    majorMeans = []
    print("Threshold")
    print(dataPixels)

    for maskImg in masks :
        mean = cv.mean(img, maskImg)
        means.append(mean[0])
        majorMeans.append(0)

    pixels = 0
    for m in (sortedMaskSize) :
        pixels += m[1]
        majorMeans[m[0]] = means[m[0]]
        if pixels > dataPixels :
            break

    results = []
    for i in range(0, classes) :
        results.append(np.zeros((height, width)))

    for x in range(width):
        for y in range(height):
             res[y,x] = majorMeans[mapping[y, x]]
             results[mapping[y, x]][y, x] = diffImg[y, x]
    # resLocal = np.zeros(res.shape, np.uint8)
    # n = 30
    # for x in range(width):
    #     for y in range(height):
    #         currentMask = np.zeros(res.shape, np.uint8)
    #         roiX = 0
    #         roiY = 0
    #         roiWidth =0
    #         roiHeight = 0
    #         if x < n :
    #             roiX = 0
    #             roiWidth = n + x
    #         else :
    #             roiX = x - n
    #             if x + n > width :
    #                 roiWidth = n + (width - x)
    #             else :
    #                 roiWidth = 2 * n
    #
    #         if y < n :
    #             roiY = 0
    #             roiHeight = y + n
    #         else :
    #             roiY = y - n
    #             if y + n > height :
    #                 roiHeight = n + (height - y)
    #             else :
    #                 roiHeight = 2 * n
    #         truncated = masks[mapping[y, x]][roiY:roiY+roiHeight, roiX:roiX+roiWidth]
    #         currentMask[roiY:roiY+roiHeight, roiX:roiX+roiWidth] = truncated
    #         # cv.imshow("truncated", truncated)
    #         # cv.imshow("current Mask", currentMask)
    #         # cv.waitKey(0)
    #         # cv.destroyAllWindows()
    #         mean = cv.mean(img, currentMask)
    #         resLocal[y, x] = mean[0]

    #
    # cv.imshow("mask", mask)
    # cv.imshow("region", resLocal)
    # cv.imshow("non-local mean", res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return res, results, equalized

def fillClasses(diffImg, classes) :
    height, width = diffImg.shape
    min, max, minLoc, maxLoc = cv.minMaxLoc(diffImg)
    thresholds = []
    masks = []
    mapping = np.zeros(diffImg.shape, np.uint8)
    maskSize = {}
    maskSize[0] = 0

    for i in range(1, classes + 1) :
        thresholds.append(calcTh(i, min, max, classes))
        masks.append(np.zeros(diffImg.shape, np.uint8))
        if i < classes :
            maskSize[i] = 0

    for x in range(width):
        for y in range(height):
            currentVal = diffImg[y, x]
            index = 0
            for th in thresholds :

                if index == 0 and currentVal < th:
                    setPixel(masks[index], x, y, 255, 1)
                    mapping[y, x] = index
                    maskSize[index] = maskSize[index] + 1
                    break

                if index > 0 and currentVal > thresholds[index-1] and currentVal <= th  :
                    setPixel(masks[index], x, y, 255, 1)
                    mapping[y, x] = index
                    maskSize[index] = maskSize[index] + 1
                    break
                index += 1

    return maskSize, masks, mapping

def calcTh(var, lMin, lMax, n) :
    return lMin + (var * (lMax-lMin)/n)

def setPixel(img, x, y, val, nb = 5) :
    img[y, x] = val
    if nb > 1 :
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

def plow(img) :
    diffImg = nonLocalGrouping(img)
    diffImg = diffImg.reshape((-1, 1))
    diffImg = np.float32(diffImg)
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 15
    ret, label, center = cv.kmeans(diffImg, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    kMeansRes = res.reshape((img.shape))

    roi = cv.selectROI("Select ROI", img)
    cv.destroyAllWindows()
    noiseImg = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    #standard deviation of noise
    std = np.std(noiseImg)
    kernel = np.zeros((2, 2))
    kernel[0][0] = 2
    kernel[1][0] = -1
    kernel[0][1] = -1
    filtered = cv.filter2D(noiseImg, -1, kernel)
    filtered = filtered.reshape((-1, 1))
    filtered = np.float32(filtered)
    filtered = filtered / math.sqrt(6)
    delMed = np.median(filtered)
    while delMed == 0 :
        min, max, minLoc, maxLoc = cv.minMaxLoc(filtered)
        x = minLoc[0]
        y = minLoc[1]
        filtered[y, x] = max + 1
        delMed = np.median(filtered)
    delMed = abs(delMed)
    # filtered = noiseImg - delMed
    med = np.median(delMed)
    sigma = 1.4826 * med
    if sigma == 0 :
        return
    h2 = 1.75 * pow(sigma, 2)
    filtred = median(img.copy(), 5)
    n = 5

    identity = pow(sigma, 2) * np.identity(n)
    height, width = img.shape

    window = (n - 1) / 2
    meanPatches = {}
    divideWith = {}
    covarianceSamples = {}
    filteredWithBorder = np.zeros([int(height + n), int(width + n)], np.uint32)
    filteredWithBorder[window:height + window, window:width + window] = img
    imgWithBorder = filteredWithBorder

    for x in range(width):
        for y in range(height):
            currX = x + window
            currY = y + window
            currentPatch = filteredWithBorder[int(currY - window) : int(currY + window + 1), int(currX - window) : int(currX + window + 1) ]
            ret, divideWithCoeffs = cv.threshold(np.float32(currentPatch), 2, 1, cv.THRESH_BINARY)
            if kMeansRes[y, x] in meanPatches :
                meanPatches[kMeansRes[y, x]] = np.add(meanPatches[kMeansRes[y, x]], currentPatch)
                divideWith[kMeansRes[y, x]] = np.add(divideWith[kMeansRes[y, x]], divideWithCoeffs)
                covHeight = -1
                covWidth = -1
                if (kMeansRes[y, x] in covarianceSamples) :
                    covHeight, covWidth = covarianceSamples[kMeansRes[y, x]].shape
                if ( covWidth < n and currX > 2 * window and currX < width - (2 * window) and currY > 2 * window and currY < height - (2 * window)) :
                    currentPatch = currentPatch.reshape((-1, 1))
                    if kMeansRes[y, x] in covarianceSamples :
                        covarianceSamples[kMeansRes[y, x]] = np.hstack((covarianceSamples[kMeansRes[y, x]], currentPatch))
                        # covariances[kMeansRes[y, x]] = np.append(covariances[kMeansRes[y, x]], currentPatch, 1)
                    else :
                        covarianceSamples[kMeansRes[y, x]]   = currentPatch
            else :
                meanPatches[kMeansRes[y, x]] = currentPatch
                divideWith[kMeansRes[y, x]] = divideWithCoeffs
                if (currX > 2 * window and currX < width - (2 * window) and currY > 2 * window and currY < height - (2 * window)) :
                    currentPatch = currentPatch.reshape((-1, 1))
                    covarianceSamples[kMeansRes[y, x]] = currentPatch

    toDel = []
    for i in covarianceSamples :
        covheight, covwidth = covarianceSamples[i].shape
        if covwidth < n :
            toDel.append(i)
    for i  in list(toDel) :
        del covarianceSamples[i]
        del meanPatches[i]

    covariances = {}
    for key in meanPatches.keys() :
        currentPatch = meanPatches[key]
        divideWithPatch = divideWith[key]
        heightPatch, widthPatch = currentPatch.shape
        for x in range(widthPatch):
            for y in range(heightPatch):
                currentPatch[y, x] = currentPatch[y, x] / divideWithPatch[y, x]
        meanPatches[key] = currentPatch

        meanOut = np.zeros(covarianceSamples[key].shape)
        covMat, test2 = cv.calcCovarMatrix(np.float32(covarianceSamples[key]), np.float32(meanOut), cv.COVAR_NORMAL | cv.COVAR_ROWS)
        covMat = covMat - identity
        eigenVals, eigenVect = la.eig(covMat)
        for x in range(eigenVals.size) :
            if eigenVals[x] < 0 :
                eigenVals[x] = 0.0001
        eigenVals = np.real(eigenVals)
        eigenVect = np.real(eigenVect)
        covariances[key] = eigenVect * eigenVals * np.transpose(eigenVect)

    resultImg = np.zeros(imgWithBorder.shape, np.uint64)
    divideWith = np.zeros(filteredWithBorder.shape, np.uint64)

    similarPatches = {}
    #half of the actual size
    searchWindow = 5
    for x in range(width):
        for y in range(height):
            currX = x + window
            currY = y + window
            currClass = kMeansRes[y, x]
            if currClass not in meanPatches :
                continue
            currentPatch = filteredWithBorder[int(currY - window) : int(currY + window + 1), int(currX - window) : int(currX + window + 1) ]


            weights = []
            patches = []
            meanPatch = np.zeros(currentPatch.shape)
            divideWithCoeffs = np.ones(currentPatch.shape)
            sumWeights = np.ones(currentPatch.shape)
            for k in range(width):
                if k < x - searchWindow :
                    continue
                if k > x + searchWindow :
                    break
                for l in range(height):
                    if l <  y - searchWindow :
                        continue
                    if l > y + searchWindow :
                        break
                    if (k != x and l != y and kMeansRes[l, k] == currClass) :
                        currK = k + window
                        currL = l + window
                        comparePatch = filteredWithBorder[int(currL - window) : int(currL + window + 1), int(currK - window) : int(currK + window + 1) ]

                        patchHeight, patchWidth = comparePatch.shape
                        for patchX in range(patchWidth):
                            for patchY in range(patchHeight):
                                if comparePatch[patchY, patchX] == 0 :
                                    comparePatch[patchY, patchX] = currentPatch[patchY, patchX]
                                elif currentPatch[patchY, patchX] == 0 :
                                    currentPatch[patchY, patchX] = comparePatch[patchY, patchX]

                        truePatch = imgWithBorder[int(currY - window) : int(currY + window + 1), int(currX - window) : int(currX + window + 1) ]
                        ret, divideWith = cv.threshold(np.float32(truePatch), 2, 1, cv.THRESH_BINARY)
                        meanPatch = meanPatch + truePatch
                        divideWithCoeffs = divideWithCoeffs + divideWith
                        weight = np.zeros(comparePatch.shape, np.float32)
                        for patchX in range(patchWidth):
                            for patchY in range(patchHeight):
                                val1 = currentPatch[patchY, patchX]
                                val1 = val1.astype(np.int32)
                                val2 = comparePatch[patchY, patchX]
                                val2 = val2.astype(np.int32)
                                test = pow(val1 - val2, 2)
                                test = math.exp(- test/ h2)
                                weight[patchY, patchX] = test
                        val = (1 / pow(sigma, 2))
                        weight = val * weight
                        # weight = weight.reshape((-1, 1))
                        weights.append(weight)
                        sumWeights = sumWeights + weight
                        patches.append(comparePatch)

            firstSum = 0
            secondSum = 0
            meanPatch = meanPatch /divideWithCoeffs
            # meanPatch = meanPatch.reshape((-1, 1))
            mat = la.inv(sumWeights * covariances[currClass] + np.identity(n))
            for i in range(len(weights)) :
                wp = (weights[i] * patches[i])
                ws = wp / sumWeights
                firstSum = firstSum + ws
                firstTerm = (weights[i] / sumWeights )
                thirdTerm = meanPatch - patches[i]
                secondSum = secondSum + firstTerm * mat * thirdTerm
            newPatch = firstSum + secondSum
            currentPatch = resultImg[int(currY - window) : int(currY + window + 1), int(currX - window) : int(currX + window + 1) ]
            newPatch = newPatch + currentPatch
            resultImg[int(currY - window) : int(currY + window + 1), int(currX - window) : int(currX + window + 1)] = newPatch
        print (x)

    resultImg = util.normalize(resultImg.copy(), 255)
    resultImg = np.uint8(resultImg)

    return resultImg

def eliminateNoise(img, noiseImg, th) :
    height, width = img.shape
    heightNoise, widthNoise = noiseImg.shape
    if heightNoise % 2 == 0 :
        heightNoise = heightNoise - 1
    if widthNoise % 2 == 0 :
        widthNoise = widthNoise - 1
    noiseImg = noiseImg[0 : heightNoise, 0 :widthNoise]
    windowX = (widthNoise - 1) / 2
    windowY = (heightNoise - 1) / 2
    smoothingMask = np.zeros(img.shape, np.float32)

    xComparePatch = windowX - 1
    for x in range(width):
        xMin = max(0, x - windowX)
        xMax = min(width, x + windowX)

        yComparePatch = windowY - 1
        for y in range(height):
            yMin = max(0, y - windowY)
            yMax = min(height, y + windowY)

            patch = img[yMin : yMax + 1, xMin : xMax + 1]
            heightPatch, widthPatch = patch.shape
            comparePatch = noiseImg[yComparePatch:yComparePatch + heightPatch, xComparePatch:xComparePatch + widthPatch]
            # l2 = patch - comparePatch
            # for m in range(widthPatch) :
            #     for n in range(heightPatch) :
            #         l2[n, m] = math.sqrt(pow(l2[n, m], 2))
            # l2 = np.mean(l2)

            # l2 = la.norm(patch - comparePatch)
            l2 = distance.euclidean(patch.reshape((-1, 1)), comparePatch.reshape((-1, 1)))
            l2 = l2 / (heightPatch * widthPatch)
            smoothingMask[y, x] = l2

            if yComparePatch > 0 :
                yComparePatch = yComparePatch - 1
            # if yMax == height - 1 :
            #     break
        if xComparePatch > 0 :
            xComparePatch = xComparePatch - 1
        # if xMax == width - 1 :
        #     break

    smoothingMask = util.normalize(smoothingMask, 255)
    smoothingMask = np.uint8(smoothingMask)
    smoothingMask = histogramOperations.equalizeHistogram(smoothingMask)
    smoothingMask = util.normalize(smoothingMask, 1)
    res = smoothingMask * img
    res = np.uint8(res)

    norm = util.normalize(img, 1.0)
    norm = np.float32(norm)
    blured = wiener(norm)
    blured = util.normalize(blured, 255)
    blured = np.uint8(blured)
    comp = np.zeros(img.shape)
    for x in range(width):
        for y in range(height):
            smootingTerm = smoothingMask[y, x]
            comp[y, x] = smootingTerm  * img[y, x] + (1 - smootingTerm) * blured[y, x]
    comp = util.normalize(comp, 255)
    comp = np.uint8(comp)

    th = np.median(smoothingMask)
    _, th = cv.threshold(smoothingMask, th, 1, cv.THRESH_BINARY)
    mix = np.zeros(img.shape)
    for x in range(width):
        for y in range(height):
            if th[y, x] == 1 :
                mix[y, x] = smootingTerm * img[y, x] + (1 - smootingTerm) * blured[y, x]
            else :
                mix[y, x] = blured[y, x]
    mix = util.normalize(mix, 255)
    mix = np.uint8(mix)

    cv.imshow("res", res)
    cv.imshow("comp", comp)
    cv.imshow("mix", mix)
    cv.imshow("blur", blured)
    cv.imshow("mask", smoothingMask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return comp

#
def filterMask(mask, window = 2, th = 3) :
    height, width = mask.shape
    result = np.zeros(mask.shape, np.float32)
    for x in range(width):
        for y in range(height):
            currentPatch = mask[max(y - window, 0):min(y + window, height), max(x - window, 0):min(x + window, width)]
            if np.sum(currentPatch) > th :
                result[y, x] = 1
    # kernel = np.ones((2,2),np.uint8)
    # result = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel)
    return result

#if average is true, th is a percent
def eliminateLineNoise(edges, th = 15, average = False ) :
    # The first cell is the number of labels
    # The second cell is the label matrix
    # The third cell is the stat matrix
    # The fourth cell is the centroid matrix
    numLabels, labels, stats, _ = cv.connectedComponentsWithStats(edges, connectivity=8)
    size = stats[1:, -1]
    result = np.zeros(edges.shape, np.uint8)
    if average:
        avg = np.mean(size)
        th = avg * th
    print th
    print("===============")
    for e in range(0, numLabels - 1):
        th_up = e + 1
        th_do = th_up

        # masking to keep only the components which meet the condition
        mask = cv.inRange(labels, th_do, th_up)
        if average:
            print(size[e])
            if size[e] > th:
                result = result + mask
        else :
            if size[e] >= th:

                # cv.imshow("mask", mask)
                # cv.waitKey(0)
                result = result + mask
    return result

def eliminateOpenStructures(img) :
    im2, contours, hierachy = cv.findContours(255 - img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    closedContoursImg = np.zeros(img.shape, np.uint8)
    openContoursImg = np.zeros(img.shape, np.uint8)
    openContours = []
    closedContours = []
    approxImgOpen = np.zeros(img.shape)
    approxImgClosed = np.zeros(img.shape)
    hierachy = hierachy[0]
    for i in range(0, len(contours)):
        if hierachy[i][2] < 0:
            openContours.append(contours[i])
            epsilon = 0.03 * cv.arcLength(contours[i], True)
            approx = cv.approxPolyDP(contours[i], epsilon, False)

            # cv.drawContours(approxImgOpen, contours[i], -1, 255, 1)
            cv.drawContours(approxImgOpen, [approx], -1, 255, 1)
            for contour in contours[i]:
                if (len(contour) < 60) :
                    for point in contour:
                        openContoursImg[point[1], point[0]] = 255
                else:
                    for point in contour:
                        closedContoursImg[point[1], point[0]] = 255
        else:
            closedContours.append(contours[i])
            epsilon = 0.03 * cv.arcLength(contours[i], True)
            approx = cv.approxPolyDP(contours[i], epsilon, False)

            # cv.drawContours(approxImgClosed,  contours[i], -1, 255, 1)
            cv.drawContours(approxImgClosed, [approx], -1, 255, 1)
            for contour in contours[i]:
                for point in contour:
                    closedContoursImg[point[1], point[0]] = 255

    im2 = closedContoursImg

    cv.imshow("closed", closedContoursImg)
    cv.imshow("open", openContoursImg)
    cv.waitKey(0)
    return im2