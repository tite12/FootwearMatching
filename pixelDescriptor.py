import numpy as np
import cv2 as cv
from numpy.linalg import norm

def threeLayeredLearning(images, masks, useHOG = False) :
    descriptors = []
    for i in range(len(images)) :
        print("first Layer")
        #Thehistogram f!i of the original pattern sets of interest
        #of eachtrainingimage xi, andthethresholdparameter n to
        #determine the proportions of dominant patterns selected from
        #each training image.
        descriptor = getDominantDescriptors(images[i], masks[i], useHOG)
        name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/' + str(i) + '_SIFT.txt'
        if useHOG :
            name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/' + str(
                i) + '_HOG.txt'
        np.savetxt(name, descriptor, delimiter=',')
        descriptors.append(descriptor)
    #Dominantpatternsets J1, J2,y, Jnj of nj images belonging to class j obtained from Algorithm 1.
    print("second Layer")
    descriptors = getDiscriminativeFeatures(descriptors)
    name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_SIFT.txt'
    if useHOG :
        name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_HOG.txt'
    np.savetxt(name, descriptors, delimiter=',')
    #Thediscriminativedominantpatternset JCj for each class j  obtained from Algorithm 2.
    #since we only have one class we dont need this step
    getGlobalFeatures()
    return descriptors

def getDominantDescriptors(img, mask, useHOG = False, winX = 20, cellX = 10, winY = 20, cellY = 10) :
    if useHOG :
        winSize = (winX, winY)
        blockSize = (cellX * 2, cellY * 2)
        blockStride = (cellX, cellY)
        cellSize = (cellX, cellY)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        useSignedGradients = True
        hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
        # hist = hog.compute(img)

        features = []
        height, width = img.shape
        for x in range(width):
            for y in range(height):
                if mask[y, x] == 0 :
                    continue
                if y - winY < 0 or y + winY > height or x - winX < 0 or x + winX > width :
                    continue
                currentPatch = img[(y - winY):(y + winY), (x - winX):(x + winX)]
                hist = hog.compute(currentPatch)
                minVal = np.amin(hist)
                maxVal = np.amax(hist)
                if minVal != 0 or maxVal != 0 :
                    features.append(hist.flatten())

        descriptors = {}
        for feature in features:
            keyFound = False
            for key in descriptors:
                distance = np.linalg.norm(np.asarray(key) - feature)
                print distance
                if distance < 1 :
                    keyFound = True
                    descriptors[key] = descriptors[key] + 1
            if not keyFound:
                tup = tuple(feature)
                descriptors[tup] = 1
        return
    else :
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, mask)
        des1 = des[0:1]
        des2 = des[1:len(des)]
        bf = cv.BFMatcher()
        descriptors= {}
        # while len(des2) > 0:
        height, width = des.shape
        for y in range(height):
            des1 = des[y:y+1]
            matches = bf.knnMatch(des1, des, k=100)
            matches = matches[0]
            tup = tuple(des1[0])
            sum = 0
            keypointsToDelete = []
            for i in range(len(matches) - 1):
                m = matches[i]
                # print(m.distance)
                # n = matches[i + 1]
                # if m.distance < 0.75 * n.distance:
                if m.distance < 400 :
                    if m.distance > 0:
                        sum = sum + 1
                        keypointsToDelete.append(m.trainIdx)
                else :
                    break
            descriptors[tup] = sum
            # if len(keypointsToDelete) > 0 :
            #     keypointsToDelete = np.sort(keypointsToDelete)
            #     des2 = np.delete(des2, keypointsToDelete, 0)
            # des1 = des2[0:1]
            # des2 = des2[1:len(des2)]

        discriminativeDescriptors = []
        for key in descriptors :
            if descriptors[key] > 0 :
                discriminativeDescriptors.append(key)

        return discriminativeDescriptors

def getDiscriminativeFeatures(features) :
    bf = cv.BFMatcher()
    discriminativeFeatures = features[0]
    for i in range(1, len(features)) :
        f1 = np.asarray(discriminativeFeatures)
        f2 = np.asarray(features[i])
        matches = bf.knnMatch(f1, f2, k=1)
        intersection = []
        for match in matches :
            if match[0].distance < 450 :
                if tuple(f2[match[0].trainIdx]) not in intersection :
                    intersection.append(tuple(f2[match[0].trainIdx]))
        discriminativeFeatures = intersection
        print("test")

    return discriminativeFeatures
    # discriminativePatterns = {}
    # for currentPatterns in patterns :
    #     if discriminativePatterns.__len__() == 0 :
    #         print("empty")
    #         discriminativePatterns = currentPatterns
    #         continue
    #     intersection = []
    #     for currentPattern in currentPatterns :
    #         for discPattern in discriminativePatterns :
    #             if cv.compareHist(np.asarray(discPattern), np.asarray(currentPattern), cv.HISTCMP_CORREL) > 0.9 :
    #                 intersection.append(discPattern)
    #                 break
    #     discriminativePatterns = intersection
    #     #TODO: check what happens if intersection is empty
    # return discriminativePatterns

def getGlobalFeatures() :
    return

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bin_n = 16 # Number of bins
    bin = np.int32(bin_n*ang/(2*np.pi))

    bin_cells = []
    mag_cells = []

    cellx = celly = 8

    for i in range(0,img.shape[0]/celly):
        for j in range(0,img.shape[1]/cellx):
            bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
            mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    return hist