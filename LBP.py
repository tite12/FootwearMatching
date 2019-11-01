from skimage import feature
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import histogramOperations
import util

def pixelToPatch(img, r) :
    s = r
    if r % 2 == 0 :
        s = s + 1
    height, width = img.shape
    res = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            currentPatch = img[max(0, y - s):min(y + s + 1, height), max(0, x - s):min(x + s + 1, width)]
            res[y, x] = np.mean(currentPatch)
    return res

def basicLBP(img, points, radius) :

    #test "uniform" and "var"
    lbp = feature.local_binary_pattern(img, points, radius, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), np.arange(0, points + 3), range(0, points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

def classify(img, window, points, radius, usePtP = False) :
    if usePtP :
        img = pixelToPatch(img, radius)
        points = min(8, points)
    height, width = img.shape
    lbpImg = np.empty((height, width, points + 2))
    for x in range(width):
        for y in range(height):
            # if x > window and y > window and x < width - window and y < height - window :
            # currentPatch = img[y-window:y+window, x-window:x+window]
            currentPatch = img[max(0, y-window):min(y+window, height), max(0, x-window):min(x+window, width)]
            hist = basicLBP(currentPatch, points, radius)
            lbpImg[y, x] = hist
        print x

    lbpImg = lbpImg * 255
    # kMeans = KMeans().fit(lbpImg)
    lbpImg = np.float32(lbpImg)
    test = img.reshape((-1, 1))
    test = np.float32(test)
    lbpImg = lbpImg.reshape((-1, points + 2))
    anotherTest = lbpImg[:, 0:1]
    # # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret, label, center = cv.kmeans(lbpImg, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    colors = np.empty((K, 3))
    for i in range(K) :
        colors[i, : ] = util.generateColor(i)
    res = colors[label.flatten()]
    # res = res[:, 0:3]
    kMeansRes = res.reshape((height, width, 3))
    kMeansRes = np.uint8(kMeansRes)

    return kMeansRes

def eliminateNoise(noiseX, noiseY, window, points, radius, img) :
    height, width = img.shape
    noisePatch = img[max(0, noiseY - window):min(noiseY + window, height), max(0, noiseX - window):min(noiseX + window, width)]
    noiseHist = basicLBP(noisePatch, points, radius)
    noiseHist = np.float32(noiseHist)


    lbpmask = np.empty((height, width), np.float32)
    lbpmaskChi = np.empty((height, width), np.float32)
    lbpmaskInt = np.empty((height, width), np.float32)
    lbpmaskBha = np.empty((height, width), np.float32)
    for x in range(width):
        for y in range(height):
            # if x > window and y > window and x < width - window and y < height - window :
            # currentPatch = img[y-window:y+window, x-window:x+window]
            currentPatch = img[max(0, y - window):min(y + window, height), max(0, x - window):min(x + window, width)]
            hist = basicLBP(currentPatch, points, radius)
            comp = cv.compareHist(noiseHist, np.float32(hist), cv.HISTCMP_CORREL)
            compChi = cv.compareHist(noiseHist, np.float32(hist), cv.HISTCMP_CHISQR)
            compInt = cv.compareHist(noiseHist, np.float32(hist), cv.HISTCMP_INTERSECT)
            compBha = cv.compareHist(noiseHist, np.float32(hist), cv.HISTCMP_BHATTACHARYYA)
            lbpmask[y, x] = comp
            lbpmaskChi[y, x] = compChi
            lbpmaskInt[y, x] = compInt
            lbpmaskBha[y, x] = compBha
        print x

    equalizedMask = util.normalize(lbpmask, 1)
    equalizedMaskChi = util.normalize(lbpmaskChi, 1)
    equalizedMaskInt = util.normalize(lbpmaskInt, 1)
    equalizedMaskBha = util.normalize(lbpmaskBha, 1)

    cv.imshow("chi", equalizedMaskChi)
    cv.imshow("Int", equalizedMaskInt)
    cv.imshow("Bha", equalizedMaskBha)
    cv.imshow("corr", equalizedMask)

    # cv.imshow("hm", np.uint8(img * 1-equalizedMask))
    # cv.imshow("im", np.uint8(img * (1-equalizedMaskInt)))
    # cv.imshow("bm", np.uint8(img * equalizedMask))
    # cv.imshow("cm", np.uint8(img * (1 - equalizedMask)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    return (1-equalizedMask), equalizedMaskChi, (1 - equalizedMaskInt), equalizedMaskBha