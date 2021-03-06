# from cv2.cv2 import line_descriptor_BinaryDescriptor
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

def getlbpimage(img, window, points, radius, usePtP = False) :
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
    height, width = img.shape
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

def eliminateNoise(noiseX, noiseY, noiseWidth, noiseHeight, window, points, radius, img) :
    height, width = img.shape
    # noisePatch = img[max(0, noiseY - window):min(noiseY + window, height), max(0, noiseX - window):min(noiseX + window, width)]
    # noiseHist = basicLBP(noisePatch, points, radius)
    if noiseWidth % 2 == 1 :
        noiseWidth = noiseWidth - 1
    noiseWidth = noiseWidth / 2
    if noiseHeight % 2 == 1 :
        noiseHeight = noiseHeight - 1
    noiseHeight = noiseHeight / 2
    noiseUL = img[noiseY:(noiseY + noiseHeight), noiseX:(noiseX + noiseWidth)]
    noiseUR = img[noiseY:(noiseY + noiseHeight), (noiseX + noiseWidth): (noiseX + noiseWidth + noiseWidth)]
    noiseLL = img[(noiseY + noiseHeight):(noiseY + noiseHeight + noiseHeight), noiseX:(noiseX + noiseWidth)]
    noiseLR = img[(noiseY + noiseHeight):(noiseY + noiseHeight + noiseHeight), (noiseX + noiseWidth):(noiseX + noiseWidth + noiseWidth)]
    noiseHistUL = basicLBP(noiseUL, points, radius)
    noiseHistUR = basicLBP(noiseUR, points, radius)
    noiseHistLL = basicLBP(noiseLL, points, radius)
    noiseHistLR = basicLBP(noiseLR, points, radius)
    histH = 500
    cv.normalize(noiseHistUL, noiseHistUL, 0, histH, cv.NORM_MINMAX)
    noiseHistUL = np.float32(noiseHistUL)
    cv.normalize(noiseHistUR, noiseHistUR, 0, histH, cv.NORM_MINMAX)
    noiseHistUR = np.float32(noiseHistUR)
    cv.normalize(noiseHistLL, noiseHistLL, 0, histH, cv.NORM_MINMAX)
    noiseHistLL = np.float32(noiseHistLL)
    cv.normalize(noiseHistLR, noiseHistLR, 0, histH, cv.NORM_MINMAX)
    noiseHistLR = np.float32(noiseHistLR)


    lbpmask = np.empty((height, width), np.float32)
    lbpmaskChi = np.empty((height, width), np.float32)
    lbpmaskInt = np.empty((height, width), np.float32)
    lbpmaskBha = np.empty((height, width), np.float32)
    lbpImg = np.empty((height, width, points + 2))
    for x in range(width):
        for y in range(height):
            # if x > window and y > window and x < width - window and y < height - window :
            # currentPatch = img[y-window:y+window, x-window:x+window]
            currentPatch = img[max(0, y - noiseHeight):min(y + noiseHeight, height), max(0, x - noiseWidth):min(x + noiseWidth, width)]
            hist = basicLBP(currentPatch, points, radius)
            lbpImg[y, x] = hist
            cv.normalize(hist, hist, 0, histH, cv.NORM_MINMAX)

            comp = max(max(cv.compareHist(noiseHistUL, np.float32(hist), cv.HISTCMP_CORREL), cv.compareHist(noiseHistUR, np.float32(hist), cv.HISTCMP_CORREL)),
                       max(cv.compareHist(noiseHistLL, np.float32(hist), cv.HISTCMP_CORREL), cv.compareHist(noiseHistLR, np.float32(hist), cv.HISTCMP_CORREL)))
            compChi = max(max(cv.compareHist(noiseHistUL, np.float32(hist), cv.HISTCMP_CHISQR),
                           cv.compareHist(noiseHistUR, np.float32(hist), cv.HISTCMP_CHISQR)),
                       max(cv.compareHist(noiseHistLL, np.float32(hist), cv.HISTCMP_CHISQR),
                           cv.compareHist(noiseHistLR, np.float32(hist), cv.HISTCMP_CHISQR)))
            compInt = max(max(cv.compareHist(noiseHistUL, np.float32(hist), cv.HISTCMP_INTERSECT),
                           cv.compareHist(noiseHistUR, np.float32(hist), cv.HISTCMP_INTERSECT)),
                       max(cv.compareHist(noiseHistLL, np.float32(hist), cv.HISTCMP_INTERSECT),
                           cv.compareHist(noiseHistLR, np.float32(hist), cv.HISTCMP_INTERSECT)))
            compBha = max(max(cv.compareHist(noiseHistUL, np.float32(hist), cv.HISTCMP_BHATTACHARYYA),
                           cv.compareHist(noiseHistUR, np.float32(hist), cv.HISTCMP_BHATTACHARYYA)),
                       max(cv.compareHist(noiseHistLL, np.float32(hist), cv.HISTCMP_BHATTACHARYYA),
                           cv.compareHist(noiseHistLR, np.float32(hist), cv.HISTCMP_BHATTACHARYYA)))
            # compChi = cv.compareHist(noiseHist, np.float32(hist), cv.HISTCMP_CHISQR)
            # compInt = cv.compareHist(noiseHist, np.float32(hist), cv.HISTCMP_INTERSECT)
            # compBha = cv.compareHist(noiseHist, np.float32(hist), cv.HISTCMP_BHATTACHARYYA)
            lbpmask[y, x] = comp
            lbpmaskChi[y, x] = compChi
            lbpmaskInt[y, x] = compInt
            lbpmaskBha[y, x] = compBha
        print x

    equalizedMask = util.normalize(lbpmask, 1) * 255
    equalizedMaskChi = util.normalize(lbpmaskChi, 1) * 255
    equalizedMaskInt = util.normalize(lbpmaskInt, 1) * 255
    equalizedMaskBha = util.normalize(lbpmaskBha, 1) * 255
    equalizedMask = np.uint8(equalizedMask)
    equalizedMaskChi = np.uint8(equalizedMaskChi)
    equalizedMaskInt = np.uint8(equalizedMaskInt)
    equalizedMaskBha = np.uint8(equalizedMaskBha)

    equalizedMask = histogramOperations.equalizeHistogram(equalizedMask)
    equalizedMaskChi = histogramOperations.equalizeHistogram(equalizedMaskChi)
    equalizedMaskInt = histogramOperations.equalizeHistogram(equalizedMaskInt)
    equalizedMaskBha = histogramOperations.equalizeHistogram(equalizedMaskBha)
    equalizedMask = np.float32(equalizedMask) / 255
    equalizedMaskChi = np.float32(equalizedMaskChi) / 255
    equalizedMaskInt = np.float32(equalizedMaskInt) / 255
    equalizedMaskBha = np.float32(equalizedMaskBha) / 255
    # for the Correlation and Intersection methods, the higher the metric, the more accurate the match
    #white means data, black means noise
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

    return (1-equalizedMask), (equalizedMaskChi), (1-equalizedMaskInt), (equalizedMaskBha), lbpImg

def threeLayeredLearning(images, masks) :
    patterns = []
    for i in range(len(images)) :
        print("first Layer")
        #Thehistogram f!i of the original pattern sets of interest
        #of eachtrainingimage xi, andthethresholdparameter n to
        #determine the proportions of dominant patterns selected from
        #each training image.
        pattern = getDominantPatterns(images[i], masks[i])
        name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/' + str(i) + '_6_24_5.txt'
        np.savetxt(name, pattern, delimiter=',')
        patterns.append(pattern)
    #Dominantpatternsets J1, J2,y, Jnj of nj images belonging to class j obtained from Algorithm 1.
    print("second Layer")
    patterns = getDiscriminativePatterns(patterns)
    name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_6_24_5.txt'
    np.savetxt(name, patterns, delimiter=',')
    #Thediscriminativedominantpatternset JCj for each class j  obtained from Algorithm 2.
    #since we only have one class we dont need this step
    getGlobalPatterns()
    return patterns

def getDominantPatterns(img, mask, th = 1, window = 6, points = 24, radius = 5) :
    height, width = img.shape
    patterns = {}
    noises = {}
    for x in range(width):
        for y in range(height):
            # if x > window and y > window and x < width - window and y < height - window :
            # currentPatch = img[y-window:y+window, x-window:x+window]
            currentPatch = img[max(0, y - window):min(y + window, height), max(0, x - window):min(x + window, width)]
            hist = np.float32(basicLBP(currentPatch, points, radius))
            tup = tuple(hist)
            histogramFound = False
            if mask[y, x] == 0 :
                for key in noises:
                    # val = cv.compareHist(hist, np.asarray(key), cv.HISTCMP_CORREL)
                    # print(val)
                    if cv.compareHist(hist, np.asarray(key), cv.HISTCMP_CORREL) > 0.9:
                        noises[key] = noises[key] + 1
                        histogramFound = True
                        break
                if not histogramFound:
                    # print("test")
                    noises[tup] = 1
            else :
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
    allPixels = float(mask.sum())

    sortedNoises = sorted(noises.items(), key = operator.itemgetter(1))
    noiseTh = 0.5
    prevSum = 0
    noisePixels = width * height
    for currentPattern in reversed(sortedNoises) :
        if currentPattern[1] < 100 :
            break
        for key in patterns:
            val = cv.compareHist(np.asarray(currentPattern[0]), np.asarray(key), cv.HISTCMP_CORREL)
            # print(val)
            if val > 0.9:
                del patterns[key]
                print("del")
                break

    sortedPatterns = sorted(patterns.items(), key = operator.itemgetter(1))
    prevSum = 0
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
            print("empty")
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