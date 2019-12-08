from _ast import keyword

import numpy as np
import cv2 as cv
from numpy import result_type
from matplotlib import pyplot as plt
import glob
import operator

import doThresholding
import histogramOperations
import filters
import util
import enhancement
import LBP
import pixelDescriptor
import signalTransform


firstVersionPreprocessing = False
LBPdenoising = False
LBPLearning = False
mainPipeline = True
SIFTdescriptor = False
HOGdescriptor = False
lbpImg = np.empty((0, 0))
signal = False
signalLearning = False
edgeDetection = False
processFiltered = False
siftMatchFiles = False

surfMatchFiles = False


lbpMatchFiles = False
signalMatchFiles = False
denseSiftMatchFiles = False
denseSurfMatchFiles = False

# lbpMatchFiles = False
# signalMatchFiles = True
# denseSiftMatchFiles = False
# denseSurfMatchFiles = False

def click_and_show(event, x, y, flags, param) :
    global lbpImg

    if event == cv.EVENT_LBUTTONDOWN:
        hist = lbpImg[y, x] * 255
        hist = np.uint8(hist)
        plt.plot(hist)
        plt.show()

if HOGdescriptor or SIFTdescriptor or LBPLearning or signalLearning :
    img3 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00003.png', 0)
    img9 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00009.jpg', 0)
    img17 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00017.jpg', 0)
    img20 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00020.jpg', 0)
    img21 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00021.jpg', 0)
    img25 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00025.jpg', 0)
    img66 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00066.jpg', 0)

    mask3 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00003_mask.jpg', 0)
    _, mask3 = cv.threshold(mask3, 125, 1, cv.THRESH_BINARY_INV)
    mask9 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00009_mask.jpg', 0)
    _, mask9 = cv.threshold(mask9, 125, 1, cv.THRESH_BINARY_INV)
    mask17 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00017_mask.jpg', 0)
    _, mask17 = cv.threshold(mask17, 125, 1, cv.THRESH_BINARY_INV)
    mask20 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00020_mask.jpg', 0)
    _, mask20 = cv.threshold(mask20, 125, 1, cv.THRESH_BINARY_INV)
    mask21 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00021_mask.jpg', 0)
    _, mask21 = cv.threshold(mask21, 125, 1, cv.THRESH_BINARY_INV)
    mask25 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00025_mask.jpg', 0)
    _, mask25 = cv.threshold(mask25, 125, 1, cv.THRESH_BINARY_INV)
    mask66 = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00066_mask.jpg', 0)
    _, mask66 = cv.threshold(mask66, 125, 1, cv.THRESH_BINARY_INV)
    # mask66 =  np.float32(mask66)
    # cv.imshow("mask", mask66)
    # cv.waitKey(0)

    test = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/output_00025_FM.jpg', 0)

    # cv.imshow("orig", img25)
    # cv.imshow("mask", test)
    # th = doThresholding.otsuThreshold(test)
    # cv.imshow("th", th)
    # cv.imshow("diff", img25 - test)
    # diffTh = np.float32(img25) - np.float32(th)
    # diffTh = util.normalize(diffTh, 1)
    # cv.imshow("diffTh", diffTh)
    # cv.waitKey(0)

    masks = [mask66, mask3, img9, mask17, mask20, mask21, mask25]

    images = [img66, img3, img9, img17, img20, img21, img25]

    if HOGdescriptor :
        # descriptors = pixelDescriptor.threeLayeredLearning(images, masks, SIFTdescriptor)
        descriptors = np.loadtxt('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_HOG.txt', delimiter=',')
        descriptors = np.float32(np.asarray(descriptors))
        img = img66
        winX = 20
        winY = 20
        hog = pixelDescriptor.getHOGDescriptor()
        height, width = img.shape
        res = np.zeros((height, width), np.float32)
        for x in xrange(0, width, 3):
            for y in xrange(0, height, 3):
                if y - winY < 0 or y + winY > height or x - winX < 0 or x + winX > width:
                    continue
                currentPatch = img[(y - winY):(y + winY), (x - winX):(x + winX)]
                hist = hog.compute(currentPatch)
                maxVal = 0
                for descriptor in descriptors:
                    val = np.linalg.norm(hist - descriptor)
                    if maxVal < val :
                        maxVal = val
                res[y, x] = maxVal
        res = util.normalize(res, 1)
        cv.imshow("res", 1 - res)
        cv.waitKey(0)
        cv.destroyAllWindows()

    if SIFTdescriptor :
        # descriptors = pixelDescriptor.threeLayeredLearning(images, masks, False)
        descriptors = np.loadtxt('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_SIFT.txt', delimiter=',')
        descriptors = np.float32(np.asarray(descriptors))
        img = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/hard/00174.jpg', 0)
        height, width = img.shape
        kp = []
        for y in range(height):
            for x in range(width):
                kp.append(cv.KeyPoint(y = float(y), x = float(x), _size = float(1)))
        sift = cv.xfeatures2d.SIFT_create()
        # kp, des = sift.detectAndCompute(img, mask)
        des = sift.compute(img, kp)
        kp = des[0]
        des = des[1]

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des,descriptors,k=100)
        keypoints2 = []
        results = np.zeros(img.shape, np.float32)
        for match in matches :
            if match[0].distance < 200 :
                keypoint = kp[match[0].queryIdx]
                keypoints2.append(keypoint)
                y = int(keypoint.pt[1])
                x = int(keypoint.pt[0])
                results[y, x] = 1
        results = filters.filterMask(results)
        output = cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        outputMatched = cv.drawKeypoints(img, keypoints2, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/SIFT/output_SIFT_matched_00197.jpg',np.uint8(results * 255))
        cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/SIFT/output_SIFT_matched_00197_alternative.jpg',outputMatched)
        cv.imshow("keypoint mask", results)
        cv.imshow("matched", outputMatched)
        cv.waitKey(0)
        cv.destroyAllWindows()

    if LBPLearning :
        # patterns = LBP.threeLayeredLearning(images, masks)
        patterns = np.loadtxt('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_4_12_3.txt', delimiter=',')
        img = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/hard/00197.jpg', 0)
        lbpImage = LBP.getLBPImage(img, 4, 12, 3)
        height, width = img.shape
        res = np.zeros((height, width), np.float32)
        print("calculating")
        for x in range(width):
            for y in range(height):
                currentHisogram = lbpImage[y, x]
                currentHisogram = np.float32(currentHisogram)
                maxVal = 0
                for pattern in patterns :
                    val = cv.compareHist(currentHisogram, np.float32(np.asarray(pattern)), cv.HISTCMP_CORREL)
                    # if val > 0.75:
                    #     res[y, x] = 1
                    #     continue
                    if maxVal < val :
                        maxVal = val
                res[y, x] = maxVal
        cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/LBP/output_00197_4_12_3.jpg', res * 255)
        cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/LBP/output_00197_4_12_3_diff.jpg', img - np.uint8((1 - res) * 255))
        cv.imshow("res", res)
        cv.imshow("res2", img - np.uint8((1 - res) * 255))
        cv.waitKey(0)

    if signalLearning :
        # features = signalTransform.threeLayeredLearning(images, masks)
        features = np.loadtxt(
            'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_FM.txt',
            delimiter=',')
        img = img21.copy()
        windowHeight = 5
        windowWidth = 5
        rep = cv.copyMakeBorder(img, windowHeight, windowHeight, windowWidth, windowWidth, cv.BORDER_REFLECT101)
        height, width = img.shape
        res = np.zeros((height, width), np.float32)
        print("calculating")
        for x in range(width):
            xInd = x + windowWidth
            for y in range(height):
                yInd = y + windowHeight
                currentPatch = rep[yInd - windowHeight:yInd + windowHeight, xInd - windowWidth:xInd + windowWidth]
                imgFM = signalTransform.calculateFourierMellin(currentPatch)
                imgMean = np.mean(imgFM)
                imgFM = imgFM - imgMean
                imgFM = imgFM.flatten()
                maxVal = 0
                for feature in features:
                    corr = corr = signalTransform.correlation(imgFM, feature)
                    if maxVal < corr:
                        maxVal = corr
                res[y, x] = maxVal
            print x
        res = util.normalize(res, 1)
        cv.imwrite(
            'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/FM/output_00021_FM_window5.jpg',
            res * 255)
        cv.imshow("res", res)
        cv.imshow("res2", img - np.uint8((1 - res) * 255))
        cv.waitKey(0)

img = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/images/orig/00233.jpg', 0)
# imgGT = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/00003.png', 0)

# roi = cv.selectROI("Select noise area", img)

# noiseImg = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

mask = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/00233_noise.jpg', 0)
mask = doThresholding.otsuThreshold(mask) / 255
# mask = np.zeros(img.shape, np.uint8)

files = []
path = "C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/GT/backup/*.png"
files = glob.glob(path)

if lbpMatchFiles :
    print("LBP match files")
    window = 5

    bf = cv.BFMatcher()
    flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    img = doThresholding.otsuThreshold(img)

    height, width = img.shape

    des1 = []
    for x in range(window, width - window, 4):
        xInd = x + window
        for y in range(window, height - window, 4):
            if mask[y, x] == 1 :
                continue
            yInd = y + window
            currentPatch = img[yInd - window:yInd + window, xInd - window:xInd + window]
            imgLBP = LBP.basicLBP(currentPatch, 36, window)
            des1.append(np.float32(imgLBP.reshape((-1, 1))))

    matchesDict = {}
    flannMatchesDict = {}
    for file in files:
        # print(file)
        gt = cv.imread(file, 0)
        gt = cv.resize(gt, (width, height))
        gt = doThresholding.otsuThreshold(gt)

        height, width = gt.shape

        des2 = []
        for x in range(window, width - window, 4):
            xInd = x + window
            for y in range(window, height - window, 4):
                yInd = y + window
                currentPatch = gt[yInd - window:yInd + window, xInd - window:xInd + window]
                gtLBP = LBP.basicLBP(currentPatch, 36, window)
                des2.append(np.float32(gtLBP.reshape((-1, 1))))

        matches = bf.knnMatch(np.asarray(des1), np.asarray(des2), k=2)
        flannMatches = flann.knnMatch(np.asarray(des1), np.asarray(des2), k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]

        good = 0
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good += 1
                matchesMask[i] = [1, 0]

        matchesDict[file] = good

        good = 0
        for m, n in flannMatches:
            if m.distance < 0.8 * n.distance:
                good += 1
                matchesMask[i] = [1, 0]

        flannMatchesDict[file] = good

    sortedMatches = sorted(matchesDict.items(), key=operator.itemgetter(1))
    sortedFlannMatches = sorted(flannMatchesDict.items(), key=operator.itemgetter(1))

    print(".....................LBP BF matches...................")
    for match in sortedMatches:
        print(match[0])
        print(match[1])
        print("::::::::::::::::::::::::::")

    print(".....................LBP FLANN matches...................")
    for match in sortedFlannMatches:
        print(match[0])
        print(match[1])
        print("::::::::::::::::::::::::::")

if signalMatchFiles :
    print("Signal match files")
    window = 5

    bf = cv.BFMatcher()
    flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    img = doThresholding.otsuThreshold(img)

    height, width = img.shape
    img = cv.copyMakeBorder(img, window, window, window, window, cv.BORDER_REFLECT101)

    des1 = []
    for x in range(0, width, 4):
        xInd = x + window
        for y in range(0, height, 4):
            if mask[y, x] == 1 :
                continue
            yInd = y + window
            currentPatch = img[yInd - window:yInd + window, xInd - window:xInd + window]
            imgFM = signalTransform.calculateFourierMellin(currentPatch)
            des1.append(imgFM.reshape((-1, 1)))

    matchesDict = {}
    flannMatchesDict = {}
    for file in files:
        # print(file)
        gt = cv.imread(file, 0)
        gt = cv.resize(gt, (width, height))
        gt = doThresholding.otsuThreshold(gt)

        height, width = gt.shape
        gt = cv.copyMakeBorder(gt, window, window, window, window, cv.BORDER_REFLECT101)

        des2 = []
        for x in range(0, width, 4):
            xInd = x + window
            for y in range(0, height, 4):
                yInd = y + window
                currentPatch = gt[yInd - window:yInd + window, xInd - window:xInd + window]
                gtFM = signalTransform.calculateFourierMellin(currentPatch)
                des2.append(gtFM.reshape((-1, 1)))

        # corr = signalTransform.correlation(img, gt)
        # matchesDict[file] = corr
        matches = bf.knnMatch(np.asarray(des1), np.asarray(des2), k=2)
        flannMatches = flann.knnMatch(np.asarray(des1), np.asarray(des2), k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]

        good = 0
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good += 1
                matchesMask[i] = [1, 0]

        matchesDict[file] = good

        good = 0
        for m, n in flannMatches:
            if m.distance < 0.8 * n.distance:
                good += 1

        flannMatchesDict[file] = good

        # gt = util.normalize(gt, 255)
        # print file
        # cv.imshow("file", gt)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    sortedMatches = sorted(matchesDict.items(), key=operator.itemgetter(1))
    sortedFlannMatches = sorted(flannMatchesDict.items(), key=operator.itemgetter(1))

    print(".....................signal BF matches...................")
    for match in sortedMatches:
        print(match[0])
        print(match[1])
        print(":::::::::::::::::::::::::::")

    print(".....................signal FLANN matches...................")
    for match in sortedFlannMatches:
        print(match[0])
        print(match[1])
        print(":::::::::::::::::::::::::::")

if siftMatchFiles or denseSiftMatchFiles :
    img = doThresholding.otsuThreshold(img)

    sift = cv.xfeatures2d.SIFT_create()
    FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=6,  # 12
    #                     key_size=12,  # 20
    #                     multi_probe_level=1)  # 2
    # search_params = dict(checks=50)  # or pass empty dictionary
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    bf = cv.BFMatcher()

    matchesDict = {}
    flannMatchesDict = {}

    if siftMatchFiles :
        print("normal SIFT matching")
        kp1, des1 = sift.detectAndCompute(img, None)

        # img = cv.drawKeypoints(img, kp1, None)
        # cv.imshow("img", img)
        # cv.waitKey(0)


        des1 = np.uint8(des1)


        for file in files :
            # print("2")
            gt = cv.imread(file, 0)

            kp2, des2 = sift.detectAndCompute(gt, None)
            des2 = np.uint8(des2)
            matches = flann.knnMatch(des1, des2, k=2)

            matchesMask = [[0, 0] for i in range(len(matches))]

            good = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.95 * n.distance:
                    good += 1
                    matchesMask[i] = [1, 0]

            matchesDict[file] = good

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=cv.DrawMatchesFlags_DEFAULT)
            img3 = cv.drawMatchesKnn(img, kp1, gt, kp2, matches, None, **draw_params)
            cv.imshow("matches", img3)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print("1")
    elif denseSiftMatchFiles:
        print("dense SIFT matching")
        height, width = img.shape
        keypoints1 = []
        for y in range(0, height, 4) :
            for x in range(0, width, 4) :
                if mask[y, x] == 1 :
                    continue
                keypoint = cv.KeyPoint(y = float(y), x = float(x), _size = float(4))
                keypoints1.append(keypoint)
        des1 = sift.compute(img, keypoints1)
        keypoints1 = des1[0]
        des1 = (des1[1])

        for file in files:
            # print("2")
            gt = cv.imread(file, 0)

            height, width = gt.shape
            keypoints2 = []
            for y in range(0, height, 4):
                for x in range(0, width, 4):
                    keypoint = cv.KeyPoint(y=float(y), x=float(x), _size=float(4))
                    keypoints2.append(keypoint)
            des2 = sift.compute(gt, keypoints2)
            keypoints2 = des2[0]
            des2 = (des2[1])
            matches = bf.knnMatch(des1, des2, k=2)
            flannMatches = flann.knnMatch(des1, des2, k=2)

            matchesMask = [[0, 0] for i in range(len(matches))]
            flannMatchesMask = [[0, 0] for i in range(len(matches))]

            good = 0
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good += 1
                    matchesMask[i] = [1, 0]

            matchesDict[file] = good

            good = 0
            for i, (m, n) in enumerate(flannMatches):
                if m.distance < 0.8 * n.distance:
                    good += 1
                    flannMatchesMask[i] = [1, 0]

            flannMatchesDict[file] = good
            # print("1")

    sortedMatches = sorted(matchesDict.items(), key=operator.itemgetter(1))
    sortedFlannMatches = sorted(flannMatchesDict.items(), key=operator.itemgetter(1))

    print(".....................dense SIFT BF matches...................")
    for match in  sortedMatches :
        print(match[0])
        print(match[1])
        print(":::::::::::::::::::::::::::::::")

    print(".....................dense SIFT FLANN matches...................")
    for match in sortedFlannMatches:
        print(match[0])
        print(match[1])
        print(":::::::::::::::::::::::::::::::")

if surfMatchFiles or denseSurfMatchFiles :
    img = doThresholding.otsuThreshold(img)

    minHessian = 400
    surf = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    bf = cv.BFMatcher()

    matchesDict = {}
    bfMatchesDict = {}
    if surfMatchFiles:
        print("normal SURF matching")
        kp1, des1 = surf.detectAndCompute(img, None)

        # img = cv.drawKeypoints(img, kp1, None)
        # cv.imshow("img", img)
        # cv.waitKey(0)


        # des1 = np.uint8(des1)


        for file in files :
            print("2")
            gt = cv.imread(file, 0)
            gt = doThresholding.otsuThreshold(gt)
            kp2, des2 = surf.detectAndCompute(gt, None)
            # des2 = np.uint8(des2)
            matches = flann.knnMatch(des1, des2, k=2)

            matchesMask = []
            good = 0
            for (m, n) in matches :
                if m.distance < 0.95 * n.distance:
                    good += 1
                    matchesMask.append(m)

            matchesDict[file] = good


            print("1")
    elif denseSurfMatchFiles:
        print("dense SURF matching")
        height, width = img.shape
        keypoints1 = []
        for y in range(0, height, 4):
            for x in range(0, width, 4):
                if mask[y, x] == 1 :
                    continue
                keypoint = cv.KeyPoint(y=float(y), x=float(x), _size=float(4))
                keypoints1.append(keypoint)
        des1 = surf.compute(img, keypoints1)
        keypoints1 = des1[0]
        des1 = des1[1]

        for file in files:
            # print("2")
            gt = cv.imread(file, 0)
            gt = doThresholding.otsuThreshold(gt)
            height, width = gt.shape
            keypoints2 = []
            for y in range(0, height, 4):
                for x in range(0, width, 4):
                    keypoint = cv.KeyPoint(y=float(y), x=float(x), _size=float(4))
                    keypoints2.append(keypoint)
            des2 = surf.compute(gt, keypoints2)
            keypoints2 = des2[0]
            des2 = des2[1]
            # des2 = np.uint8(des2)
            matches = flann.knnMatch(des1, des2, k=2)
            bfMatches = bf.knnMatch(des1, des2, k=2)

            good = 0
            for (m, n) in matches:
                if m.distance < 0.8 * n.distance:
                    good += 1

            matchesDict[file] = good

            good = 0
            for (m, n) in bfMatches:
                if m.distance < 0.8 * n.distance:
                    good += 1

            bfMatchesDict[file] = good

            # print("1")

    sortedMatches = sorted(matchesDict.items(), key=operator.itemgetter(1))
    sortedBfMatches = sorted(bfMatchesDict.items(), key=operator.itemgetter(1))

    print(".....................dense SURF BF matches...................")
    for match in sortedBfMatches:
        print(match[0])
        print(match[1])
        print("::::::::::::::::::::::::")

    print(".....................dense SURF FLANN matches...................")
    for match in  sortedMatches :
        print(match[0])
        print(match[1])
        print("::::::::::::::::::::::::")

if processFiltered :
    img = doThresholding.otsuThreshold(img)
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv.dilate(img, kernel)
    # # img = doThresholding.calculateSkeleton(img)
    # cv.imshow("rees", img)
    # cv.waitKey(0)

    im2, contours, hierachy = cv.findContours(255-img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE )

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
                for point in contour:
                    openContoursImg[point[1], point[0]] = 255
        else :
            closedContours.append(contours[i])
            epsilon = 0.03 * cv.arcLength( contours[i], True)
            approx = cv.approxPolyDP( contours[i], epsilon, False)

            # cv.drawContours(approxImgClosed,  contours[i], -1, 255, 1)
            cv.drawContours(approxImgClosed, [approx], -1, 255, 1)
            for contour in contours[i] :
                for point in contour:
                    closedContoursImg[point[1], point[0]] = 255




    im2 = closedContoursImg + openContoursImg
    # openContoursImg = filters.eliminateLineNoise(openContoursImg, 100)
    # closedContoursImg = filters.eliminateLineNoise(closedContoursImg, 0.9, True)
    cv.imshow("rees", closedContoursImg + openContoursImg)
    cv.imshow("orig",im2)
    cv.imshow("img", img)
    cv.imshow("open", openContoursImg)
    cv.imshow("close", closedContoursImg)
    cv.imshow("approx open", approxImgOpen)
    cv.imshow("approx closed", approxImgClosed)
    cv.imshow("approx added", approxImgClosed + approxImgOpen)
    cv.waitKey(0)



if signal:
    # test = np.repeat(img, 2, axis=2)
    corr = signalTransform.eliminateNoise(img, roi[0], roi[1], 106, False )
    corr = np.uint8(util.normalize(corr, 255))
    # cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/easy/00204_corr.jpg', corr)
    # corr = noiseImg
    th = doThresholding.otsuThreshold(corr)
    kernel = np.ones((8,8),np.uint8)
    morph = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
    mask = util.normalize(morph, 1)

    # img = np.uint8(diffTh)
    cv.imshow("mask", mask)
    cv.imshow("th", th)
    cv.waitKey(0)

if mainPipeline :
    img = signalTransform.eliminateNoiseOnPattern(img, mask)
    img = util.normalize(img, 255)
    img = np.uint8(img)
    img, classes, equalized = filters.regionBasedNonLocalMeans(img)

    biggestClassIndex = 0
    biggestClassCount = 0
    i = 0
    if not equalized:
        for currentClass in classes:
            currentSum = currentClass
            currentSum[currentClass > 0] = 1
            # cv.imshow("class", np.uint8(util.normalize(currentClass, 255)))
            # cv.waitKey(0)
            sum = np.sum(currentSum)
            if sum > biggestClassCount :
                biggestClassIndex = i
                biggestClassCount = sum
            i = i + 1
    result = np.zeros(img.shape)
    i = 0
    for currentClass in classes:
        if not equalized and i == biggestClassIndex :
            i = i + 1
            continue
        result = result + currentClass
        i = i + 1
    result = np.uint8(util.normalize(result, 255))
    adaptive = doThresholding.adaptiveThreshold(result)
    result[result > 0] = 1
    adaptive = adaptive * result
    adaptive = np.uint8(adaptive)

    adaptive = adaptive * (1 - mask)

    morph = filters.eliminateLineNoise(np.uint8(adaptive))
    morph = filters.eliminateLineNoise(255 - morph)


    cv.imshow("img", img)
    cv.imshow("res", adaptive)
    cv.imshow("morph", morph)
    cv.waitKey(0)
    #
    # cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/00009_extracted.jpg', adaptive)
    # cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/00009_filtered.jpg', morph)
    # cv.imwrite('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/00009_noise.jpg', mask * 255)

if edgeDetection:
    noiseImg = doThresholding.otsuThreshold(noiseImg) / 255
    kernel = np.ones((1, 1), np.uint8)
    noiseImg = cv.erode(noiseImg, kernel)

    edges = doThresholding.canny(img, noiseImg)
    kernel = np.ones((5, 5), np.uint8)
    noiseImg = cv.erode(noiseImg, kernel)
    hnEdges = doThresholding.holisticallyNestedEdgeDetection(img, noiseImg, True)

    imgGT = doThresholding.otsuThreshold(imgGT)
    hnEdgesGT = doThresholding.holisticallyNestedEdgeDetection(imgGT, np.ones(imgGT.shape))
    hnEdgesGT = np.uint8(hnEdgesGT * 255)
    edgesGT = doThresholding.canny(hnEdgesGT, np.ones(imgGT.shape))


    edges = filters.eliminateLineNoise(np.uint8(edges), 20)
    edgesGT = filters.eliminateLineNoise(np.uint8(edgesGT))

    cv.imshow("orig", img)
    cv.imshow("edges", edges)
    cv.imshow("gt", edgesGT)
    cv.imshow("HN edges", hnEdges)
    cv.imshow("HN gt", hnEdgesGT)
    cv.waitKey(0)
    cv.destroyAllWindows()

if LBPdenoising :
    # enh = enhancement.fastSMQT(img)
    corr, chi, inr, bha, lbpImg = LBP.eliminateNoise(int(roi[0]), int(roi[1]), roi[2], roi[3],  8, 24, 8, img)
    # corrI, chiI, inrI, bhaI, lbpImg = LBP.eliminateNoise(int(roi[0]), int(roi[1]), roi[2], roi[3],  8, 24, 8, inr)
    # corrC, chiC, inrC, bhaC, lbpImg = LBP.eliminateNoise(int(roi[0]), int(roi[1]), roi[2], roi[3],  8, 24, 8, corr)
    # corrCh, chiCh, inrCh, bhaCh, lbpImg = LBP.eliminateNoise(int(roi[0]), int(roi[1]), roi[2], roi[3],  8, 24, 8, chi)
    # corrB, chiB, inrB, bhaB, lbpImg = LBP.eliminateNoise(int(roi[0]), int(roi[1]), roi[2], roi[3],  8, 24, 8, bha)

    corrO = doThresholding.otsuThreshold(np.uint8(util.normalize(corr, 255)))
    chiO = doThresholding.otsuThreshold(np.uint8(util.normalize(chi, 255)))
    bhaO = doThresholding.otsuThreshold(np.uint8(util.normalize(bha, 255)))
    inrO = doThresholding.otsuThreshold(np.uint8(util.normalize(inr, 255)))

    kernel = np.ones((8, 8), np.uint8)
    corrO = cv.morphologyEx(corrO, cv.MORPH_OPEN, kernel)
    corrO = cv.morphologyEx(corrO, cv.MORPH_CLOSE, kernel)

    chiO = cv.morphologyEx(chiO, cv.MORPH_OPEN, kernel)
    chiO = cv.morphologyEx(chiO, cv.MORPH_CLOSE, kernel)

    bhaO = cv.morphologyEx(bhaO, cv.MORPH_OPEN, kernel)
    bhaO = cv.morphologyEx(bhaO, cv.MORPH_CLOSE, kernel)

    inrO = cv.morphologyEx(inrO, cv.MORPH_OPEN, kernel)
    inrO = cv.morphologyEx(inrO, cv.MORPH_CLOSE, kernel)

    cv.imwrite(
        'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/LBP/denoise/00066_corr_o.jpg',
        corrO)
    cv.imwrite(
        'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/LBP/denoise/00066_chi_o.jpg',
        chiO)
    cv.imwrite(
        'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/LBP/denoise/00066_int_o.jpg',
        inrO)
    cv.imwrite(
        'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/results/LBP/denoise/00066_bha_o.jpg',
        bhaO)


    cv.namedWindow("hist")
    cv.setMouseCallback("hist", click_and_show)
    cv.imshow("hist", img)
    cv.imshow("corr", corr)
    cv.imshow("chi", chi)
    cv.imshow("inr", inr)
    cv.imshow("bha", bha)
    cv.waitKey(0)
    cv.destroyAllWindows()


    corrMed = np.mean(corr)
    corr[corr < corrMed] = 0.0
    corr[corr >= corrMed] = 1.0
    chiMed = np.mean(chi)
    chi[chi < chiMed] = 0.0
    chi[chi >= chiMed] = 1.0
    intMed = np.mean(inr)
    inr[inr < intMed] = 0.0
    inr[inr >= intMed] = 1.0
    bhaMed = np.mean(bha)
    bha[bha < bhaMed] = 0.0
    bha[bha >= bhaMed] = 1.0



    corrImg = np.uint8(img * corr)
    chiImg = np.uint8(img * chi)
    intImg = np.uint8(img * inr)
    bhaImg = np.uint8(img * bha)
    # # img = filters.eliminateNoise(img, noiseImg, 0.1)
    #
    # # corrImg = histogramOperations.equalizeHistogram(corrImg)
    # #
    # # chiImg = histogramOperations.equalizeHistogram(chiImg)
    # #
    # # intImg = histogramOperations.equalizeHistogram(intImg)
    # #
    # # bhaImg = histogramOperations.equalizeHistogram(bhaImg)
    #
    #
    #
    #
    #


    cv.imshow("corr", corrImg)
    cv.imshow("chi", chiImg)
    cv.imshow("inr", intImg)
    cv.imshow("bha", bhaImg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # corrImg = filters.regionBasedNonLocalMeans(img, corr, True)
    # chiImg = filters.regionBasedNonLocalMeans(img, chi, True)
    # intImg = filters.regionBasedNonLocalMeans(img, inr, True)
    # bhaImg = filters.regionBasedNonLocalMeans(img, bha, True)
    #
    #
    #
    # cv.imshow("corr", corrImg)
    # cv.imshow("chi", chiImg)
    # cv.imshow("inr", intImg)
    # cv.imshow("bha", bhaImg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

if (firstVersionPreprocessing) :
    bgrImg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    norm = util.normalize(img, 1.0)
    norm = np.float32(norm)
    #==============filtering=================
    pde = filters.pde(bgrImg.copy())
    img_pde = bgrImg - pde

    epf = filters.epf(bgrImg.copy())
    img_epf = bgrImg - epf

    median5 = filters.median(img.copy(), 5)
    median9 = filters.median(img.copy(), 9)
    img_med5 = img - median5
    img_med9 = img - median9

    bi5 = filters.bilateralFilter(img.copy(), 5)
    bi9 = filters.bilateralFilter(img.copy(), 9)
    img_bi5 = img - bi5
    img_bi9 = img - bi9

    wiener = filters.wiener(norm.copy())
    img_wiener = norm - wiener
    img_wiener = util.normalize(img_wiener, 255.0)
    img_wiener = img_wiener.astype(np.uint8)

    normBi5 = util.normalize(bi5.copy(), 1.0)
    normBi5 = np.float32(normBi5)

    img_wienerBi5 = normBi5 - wiener
    img_wienerBi5 = util.normalize(img_wienerBi5.copy(), 255.0)
    img_wienerBi5 = img_wienerBi5.astype(np.uint8)

    cv.imshow("orig", img)
    cv.imshow("img-wiener", img_wiener)
    cv.imshow("wiener", wiener)
    cv.imshow("img-wienerBi5", img_wienerBi5)
    cv.imshow("img-pde", img_pde)
    cv.imshow("pde", pde)
    cv.imshow("img-epf", img_epf)
    cv.imshow("epf", epf)
    cv.imshow("img-med5", img_med5)
    cv.imshow("med5", median5)
    cv.imshow("img-med9", img_med9)
    cv.imshow("med9", median9)
    cv.imshow("img-bi5", img_bi5)
    cv.imshow("bi5", bi5)
    cv.imshow("img-bi9", img_bi9)
    cv.imshow("bi9", bi9)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #=============enhancement====================
    norm = util.normalize(img_wiener.copy(), 1.0)
    norm = np.float32(norm)
    normBi = util.normalize(img_wienerBi5.copy(), 1.0)
    normBi = np.float32(normBi)

    smqt = enhancement.fastSMQT(img_wiener.copy())
    smqtBi = enhancement.fastSMQT(img_wienerBi5.copy())

    fuzzy = enhancement.fuzzyEnhancement(norm.copy())
    fuzzyBi = enhancement.fuzzyEnhancement(normBi.copy())

    adaptive = enhancement.adaptiveEnhancement(norm.copy())
    adaptiveBi = enhancement.adaptiveEnhancement(normBi.copy())

    normFuzzy = util.normalize(fuzzy.copy(), 1.0)
    normFuzzy = np.float32(normFuzzy)
    normFuzzyBi = util.normalize(fuzzyBi.copy(), 1.0)
    normFuzzyBi = np.float32(normFuzzyBi)

    adaptiveAndFuzzy = enhancement.adaptiveEnhancement(normFuzzy.copy())
    adaptiveAndFuzzyBi = enhancement.adaptiveEnhancement(normFuzzyBi.copy())

    normSMQT = util.normalize(smqt.copy(), 1.0)
    normSMQT = np.float32(normSMQT)
    normSMQTBi = util.normalize(smqtBi.copy(), 1.0)
    normSMQTBi = np.float32(normSMQTBi)

    smqtAndFuzzy = enhancement.adaptiveEnhancement(normSMQT.copy())
    smqtAndFuzzyBi = enhancement.adaptiveEnhancement(normSMQTBi.copy())

    hist = histogramOperations.equalizeHistogram(img_wiener.copy())
    histBi = histogramOperations.equalizeHistogram(img_wienerBi5.copy())

    normFuzzy = util.normalize(fuzzy.copy(), 255)
    normFuzzy = normFuzzy.astype(np.uint8)
    normFuzzyBi = util.normalize(fuzzyBi.copy(), 255)
    normFuzzyBi = normFuzzyBi.astype(np.uint8)

    histFuzzy = histogramOperations.equalizeHistogram(normFuzzy.copy())
    histFuzzyBi = histogramOperations.equalizeHistogram(normFuzzyBi.copy())

    cv.imshow("orig", img_wiener)
    cv.imshow("origBi", img_wienerBi5)
    cv.imshow("smqt", smqt)
    cv.imshow("smqtBi", smqtBi)
    cv.imshow("fuzzy", fuzzy)
    cv.imshow("fuzzyBi", fuzzyBi)
    cv.imshow("smqt and fuzzy", smqtAndFuzzy)
    cv.imshow("smqt and fuzzy Bi", smqtAndFuzzyBi)
    cv.imshow("adaptive", adaptive)
    cv.imshow("adaptiveBi", adaptiveBi)
    cv.imshow("adaptive and fuzzy", adaptiveAndFuzzy)
    cv.imshow("adaptive and fuzzy Bi", adaptiveAndFuzzyBi)
    cv.imshow("hist", hist)
    cv.imshow("histBi", histBi)
    cv.imshow("hist and fuzzy", histFuzzy)
    cv.imshow("hist and fuzzy Bi", histFuzzyBi)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #=============thresholding====================
    norm = util.normalize(smqtBi.copy(), 255.0)
    norm = norm.astype(np.uint8)
    normHist = util.normalize(histFuzzyBi.copy(), 255.0)
    normHist = normHist.astype(np.uint8)

    adaptive = doThresholding.adaptiveThreshold(smqtBi.copy())
    adaptiveHist = doThresholding.adaptiveThreshold(histFuzzyBi.copy())

    otsu = doThresholding.otsuThreshold(smqtBi.copy())
    otsuHis = doThresholding.otsuThreshold(histFuzzyBi.copy())

    niblack = doThresholding.niblackThreshold(norm.copy())
    niblackHist = doThresholding.niblackThreshold(normHist.copy())
    niblack = util.normalize(niblack, 255)
    niblackHist = util.normalize(niblackHist, 255)

    sauvola = doThresholding.niblackThreshold(norm.copy(), 1, 9, 0.025)
    sauvolaHist = doThresholding.niblackThreshold(normHist.copy(), 1, 9, 0.025)
    sauvola = util.normalize(sauvola, 255)
    sauvolaHist = util.normalize(sauvolaHist, 255)

    wolf = doThresholding.niblackThreshold(norm.copy(), 2, 9, 0.025)
    wolfHist = doThresholding.niblackThreshold(normHist.copy(), 2, 9, 0.025)
    wolf = util.normalize(wolf, 255)
    wolfHist = util.normalize(wolfHist, 255)

    nick = doThresholding.niblackThreshold(norm.copy(), 3, 9, 0.025)
    nickHist = doThresholding.niblackThreshold(normHist.copy(), 3, 9, 0.025)
    nick = util.normalize(nick, 255)
    nickHist = util.normalize(nickHist, 255)

    skel = doThresholding.calculateSkeleton(smqtBi.copy())
    skelHist = doThresholding.calculateSkeleton(histFuzzyBi.copy())

    thin = doThresholding.thinning(smqtBi.copy())
    thinHist = doThresholding.thinning(histFuzzyBi.copy())

    cv.imshow("orig", smqtBi)
    cv.imshow("orig Hist", histFuzzyBi)
    cv.imshow("adaptive th", adaptive)
    cv.imshow("adaptive Hist", adaptiveHist)
    cv.imshow("otsu", otsu)
    cv.imshow("otsu Hist", otsuHis)
    cv.imshow("niblack", niblack)
    cv.imshow("niblack hist", niblackHist)
    cv.imshow("sauvola", sauvola)
    cv.imshow("sauvola hist", sauvolaHist)
    cv.imshow("wolf", wolf)
    cv.imshow("wolf hist", wolfHist)
    cv.imshow("nick", nick)
    cv.imshow("nick hist", nickHist)
    cv.imshow("skeleton", skel)
    cv.imshow("skeleton Hist", skelHist)
    cv.imshow("thinning", thin)
    cv.imshow("thinning Hist", thinHist)
    cv.waitKey(0)
    cv.destroyAllWindows()

# canny = cv.Canny(img,100, 100)
# hist = histogramOperations.equalizeHistogram(img.copy())
# smqt = enhancement.fastSMQT(img.copy())
#
# norm = util.normalize(img, 1.0)
# norm = np.float32(norm)
#
# bin0 = doThresholding.niblackThreshold(img.copy(), 0, 25, 0.1)
# bin1 = doThresholding.niblackThreshold(img.copy(), 1, 25, 0.1)
# bin2 = doThresholding.niblackThreshold(img.copy(), 2, 25, 0.1)
# bin3 = doThresholding.niblackThreshold(img.copy(), 3, 25, 0.1)
# bin = bin.astype(np.uint8)
# bin0 = util.normalize(bin0, 255)
# bin1 = util.normalize(bin1, 255)
# bin2 = util.normalize(bin2, 255)
# bin3 = util.normalize(bin3, 255)
#
# fuzzyImg = enhancement.fuzzyEnhancement(norm.copy())
# adaptiveImg = enhancement.adaptiveEnhancement(norm.copy())
#
# wiener =filters.wiener(norm.copy())
# wiener = util.normalize(wiener, 1.0)
# wiener = 1 - wiener
# wiener = norm - wiener

# bgrImg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# pde = filters.pde(bgrImg.copy(), 5, 0.1, 0.02)
# epf = filters.epf(bgrImg, 3, 50)
#
# cv.imshow("orig", img)
# cv.imshow("hist", hist)
# cv.imshow("smqt", smqt)
# cv.imshow("fuzzy", fuzzyImg)
# cv.imshow("adaptive", adaptiveImg)
# cv.imshow("wiener", wiener)
# cv.imshow("Canny", canny)
# cv.imshow("pde", pde)
# cv.imshow("epf", epf)
# cv.imshow("diff", bgrImg - pde)
# cv.imshow("bin0", bin0)
# cv.imshow("bin1", bin1)
# cv.imshow("bin2", bin2)
# cv.imshow("bin3", bin3)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# img = np.float32(img)
# img = img * 1.0/255

# img = doThresholding.calculateSkeleton(img)
# wienerFilteredImg = filters.wiener(img.copy())
# invertedWienerFiltered = 1.0-wienerFilteredImg
# wienerImg = img - invertedWienerFiltered
#
# wienerImg = util.normalize(wienerImg, 1.0)
# wienerImg = np.float32(wienerImg)
#
# fuzzyImg = enhancement.fuzzyEnhancement(wienerImg.copy())
# adaptiveImg = enhancement.adaptiveEnhancement(fuzzyImg.copy())
#
# adaptiveImg = util.normalize(adaptiveImg, 255)
# adaptiveImg = adaptiveImg.astype(np.uint8)

# smqtImg = doThresholding.fastSMQT(adaptiveImg.copy())

# faDiff = fuzzyImg - adaptiveImg
# asdiff = adaptiveImg - smqtImg
# asdiff = util.normalize(asdiff, 255)
# asdiff = asdiff.astype(np.uint8)

# skeletonInvImg = doThresholding.thinning(255 - adaptiveImg)
# skeletonImg = doThresholding.thinning(adaptiveImg.copy())
# otsuThImg = doThresholding.otsuThreshold(adaptiveImg.copy())
# adaptiveThImg = doThresholding.adaptiveThreshold(adaptiveImg.copy())

# diffSkeletonInvImg = doThresholding.calculateSkeleton(255 - asdiff)
# diffSkeletonImg = doThresholding.calculateSkeleton(asdiff.copy())
# diffOtsuThImg = doThresholding.otsuThreshold(asdiff.copy())
# diffAdaptiveThImg = doThresholding.adaptiveThreshold(asdiff.copy())

# img = 1.0 - img
# img = histogramOperations.equalizeHistogram(img)
# cv.imshow("wiener filtered", wienerImg)
# cv.imshow("fuzzy img", fuzzyImg)
# cv.imshow("fuzzy-adaptive diff img", faDiff)
# cv.imshow("adaptive img", adaptiveImg)
# cv.imshow("adaptive SMQT diff img", asdiff)
# cv.imshow("SMQT img", smqtImg)

# cv.imshow("otsu th img", otsuThImg)
# cv.imshow("adaptive th img", adaptiveThImg)
# cv.imshow("skeleton img", skeletonImg)
# cv.imshow("skeleton Inv img", skeletonInvImg)

# cv.imshow("diff otsu th img", diffSkeletonInvImg)
# cv.imshow("diff adaptive th img", diffAdaptiveThImg)
# cv.imshow("diff skeleton img", diffSkeletonImg)
# cv.imshow("diff skeleton Inv img", diffSkeletonInvImg)

# cv.waitKey(0)
# cv.destroyAllWindows()

# doThresholding.fastSMQT(img)
#
# img = np.float32(img)
# img = img * 1.0/255
# orig = img.copy();
# # returns img in format uint8
# img = doThresholding.SMQT(img)

# cv.imshow("orig",img)
# img = doThresholding.fuzzyEnhancement(img)
# cv.imshow("fuzzy",img)
# img = doThresholding.adaptiveEnhancement(img)
# cv.imshow("adaptive",img)
#
# img = img * 255
# img = img.astype(np.uint8)
#
# for x in range(1):
#     img = doThresholding.blur(img)
#     img = doThresholding.adaptiveThreshold(img)
#     cv.imshow("thresholding", img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

##DoG
# g1Img = cv.GaussianBlur(img, (1, 1), 0)
# g2Inf = cv.GaussianBlur(img, (3, 3), 0)
# img = g1Img - g2Inf
# img = (255-img)
# cv.imshow("g1",g1Img)
# cv.imshow("g2",g2Inf)

#morph
# kernel = np.ones((2,2),np.uint8)
# img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)


# cv.imshow("img",img)
# cv.imshow("orig",orig)
# cv.waitKey(0)
# cv.destroyAllWindows()

