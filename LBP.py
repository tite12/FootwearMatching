from skimage import feature
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import histogramOperations

def pixelToPatch(img) :

    return

def basicLBP(img, points, radius) :

    #test "uniform" and "var"
    lbp = feature.local_binary_pattern(img, points, radius, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), np.arange(0, points + 3), range(0, points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

def classifiy(img, window, points, radius) :
    height, width = img.shape
    lbpImg = np.empty((height, width, points + 2))
    for x in range(width):
        for y in range(height):
            if x > window and y > window and x < width - window and y < height - window :
                currentPatch = img[y-window:y+window, x-window:x+window]
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
    res = center[label.flatten()]
    res = res[:, 0:3]
    kMeansRes = res.reshape((height, width, 3))
    kMeansRes = cv.cvtColor(kMeansRes, cv.COLOR_BGR2GRAY)

    histogramOperations.equalizeHistogram(kMeansRes)
    cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/easy/00182color.jpg', kMeansRes)
    cv.imshow("kMeans", kMeansRes)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return