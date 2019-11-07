from cv2.cv2 import line_descriptor_BinaryDescriptor
from skimage import feature
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import histogramOperations
import util
import operator

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

def getLBPImage(img, window, points, radius, usePtP = False) :
    if usePtP:
        img = pixelToPatch(img, radius)
        points = min(8, points)
    height, width = img.shape
    lbpImg = np.empty((height, width, points + 2))
    for x in range(width):
        for y in range(height):
            # if x > window and y > window and x < width - window and y < height - window :
            # currentPatch = img[y-window:y+window, x-window:x+window]
            currentPatch = img[max(0, y - window):min(y + window, height), max(0, x - window):min(x + window, width)]
            hist = basicLBP(currentPatch, points, radius)
            lbpImg[y, x] = hist
        print x
    return lbpImg

def classify(img, window, points, radius, usePtP = False) :
    # if usePtP :
    #     img = pixelToPatch(img, radius)
    #     points = min(8, points)
    # height, width = img.shape
    # lbpImg = np.empty((height, width, points + 2))
    # for x in range(width):
    #     for y in range(height):
    #         # if x > window and y > window and x < width - window and y < height - window :
    #         # currentPatch = img[y-window:y+window, x-window:x+window]
    #         currentPatch = img[max(0, y-window):min(y+window, height), max(0, x-window):min(x+window, width)]
    #         hist = basicLBP(currentPatch, points, radius)
    #         lbpImg[y, x] = hist
    #     print x
    lbpImg = getLBPImage(img, window, points, radius, usePtP)
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

def eliminateNoise(noiseX, noiseY, noise, window, points, radius, img) :
    height, width = img.shape
    # noisePatch = img[max(0, noiseY - window):min(noiseY + window, height), max(0, noiseX - window):min(noiseX + window, width)]
    # noiseHist = basicLBP(noisePatch, points, radius)
    noiseHist = basicLBP(noise, points, radius)
    histH = 500
    cv.normalize(noiseHist, noiseHist, 0, histH, cv.NORM_MINMAX)
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
            cv.normalize(hist, hist, 0, histH, cv.NORM_MINMAX)
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
    cv.imshow("Int", 1-equalizedMaskInt)
    cv.imshow("Bha", equalizedMaskBha)
    cv.imshow("corr", 1-equalizedMask)

    # cv.imshow("hm", np.uint8(img * 1-equalizedMask))
    # cv.imshow("im", np.uint8(img * (1-equalizedMaskInt)))
    # cv.imshow("bm", np.uint8(img * equalizedMask))
    # cv.imshow("cm", np.uint8(img * (1 - equalizedMask)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    return (1-equalizedMask), equalizedMaskChi, (1 - equalizedMaskInt), equalizedMaskBha

def threeLayeredLearning(images, masks) :
    patterns = []
    for i in range(len(images)) :
        print("first Layer")
        #Thehistogram f!i of the original pattern sets of interest
        #of eachtrainingimage xi, andthethresholdparameter n to
        #determine the proportions of dominant patterns selected from
        #each training image.
        pattern = getDominantPatterns(images[i], masks[i])
        name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/' + str(i) + '.txt'
        np.savetxt(name, pattern, delimiter=',')
        patterns.append(pattern)
    #Dominantpatternsets J1, J2,y, Jnj of nj images belonging to class j obtained from Algorithm 1.
    print("second Layer")
    patterns = getDiscriminativePatterns(patterns)
    name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative.txt'
    np.savetxt(name, patterns, delimiter=',')
    #Thediscriminativedominantpatternset JCj for each class j  obtained from Algorithm 2.
    #since we only have one class we dont need this step
    getGlobalPatterns()
    return patterns

def getDominantPatterns(img, mask, th = 1, window = 4, points = 8, radius = 3) :
    height, width = img.shape
    patterns = {}
    for x in range(width):
        for y in range(height):
            if mask[y, x] == 0 :
                continue
            # if x > window and y > window and x < width - window and y < height - window :
            # currentPatch = img[y-window:y+window, x-window:x+window]
            currentPatch = img[max(0, y - window):min(y + window, height), max(0, x - window):min(x + window, width)]
            hist = np.float32(basicLBP(currentPatch, points, radius))
            tup = tuple(hist)
            back = np.asarray(tup)
            histogramFound = False
            for key in patterns :
                # val = cv.compareHist(hist, np.asarray(key), cv.HISTCMP_CORREL)
                # print(val)
                if cv.compareHist(hist, np.asarray(key), cv.HISTCMP_CORREL) > 0.9 :
                    patterns[key] = patterns[key] + 1
                    histogramFound = True
                    break
            if not histogramFound :
                # print("test")
                patterns[tup] = 1
        print x
    sortedPatterns = sorted(patterns.items(), key = operator.itemgetter(1))
    prevSum = 0
    allPixels = float(mask.sum())
    dominantPatterns = []
    for currentPattern in reversed(sortedPatterns) :
        dominantPatterns.append(currentPattern[0])
        val = currentPattern[1] / allPixels + prevSum
        if (val > th) :
            break
        else :
            prevSum = val
    return dominantPatterns

def getDiscriminativePatterns(patterns) :
    discriminativePatterns = {}
    for currentPatterns in patterns :
        if discriminativePatterns.__len__() == 0 :
            discriminativePatterns = currentPatterns
            continue
        intersection = []
        for currentPattern in currentPatterns :
            for discPattern in discriminativePatterns :
                if cv.compareHist(np.asarray(discPattern), np.asarray(currentPattern), cv.HISTCMP_CORREL) > 0.9 :
                    intersection.append(discPattern)
                    break
        discriminativePatterns = intersection
        #TODO: check what happens if intersection is empty
    return discriminativePatterns

def getGlobalPatterns() :
    return