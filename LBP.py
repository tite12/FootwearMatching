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