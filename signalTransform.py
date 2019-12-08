import numpy as np
import cv2 as cv
import math
import scipy.ndimage.interpolation as ndii
import operator
import util
import histogramOperations
from matplotlib import pyplot as plt

# global constants
RE_IDX = 0
IM_IDX = 1
ROWS_AXIS = 0
COLS_AXIS = 1
polarMode = "spline"
noiseMode = "none" # "gaussian", "s&p", "none"
noiseIntensity = {'sigma' : 2, 'mean' : 0, 'whiteThreshold' : 0.01, 'blackThreshold' : 0.99}
resultsComparation = False

# Central point for running FFT
def calculateFft(img):
    imgTmp  = np.float32(img)
    # FFT of the image
    imgFft = cv.dft(imgTmp,flags = cv.DFT_COMPLEX_OUTPUT)
    # the FFT shift is needed in order to center the results
    imgFftShifted = np.fft.fftshift(imgFft)
    return (imgFft, imgFftShifted)

# applies highpass filter and returns the image
# H(col, row) = (1.0 - X(col, row)) * (2.0 - X(col, row)), row and col have to be transformed to range <-pi/2, pi/2>
# X(valX, valY) = cos(pi * valX) * cos(pi * valY), both valX and valY in range <-pi/2, pi/2>
def prepareHighPassFilter(img):
    pi2 = math.pi / 2.0
    # transform number of rows to <-pi/2,pi/2> range and calculate cos for each element
    rows = np.cos(np.linspace(-pi2, pi2, img.shape[0]))
    # transform number of cols to <-pi/2,pi/2> range and calculate cos for each element
    cols = np.cos(np.linspace(-pi2, pi2, img.shape[1]))
    # creates matrix the whole image
    x = np.outer( rows, cols)
    return (1.0 - x) * (2.0 - x)

# this function will calculates parameters for log polar transformation
# (center of transformation, angle step and log base)
def computeLogPolarParameters(img):
	# Step 1 - Get center of the transformation
    centerTrans = [math.floor((img.shape[ROWS_AXIS] + 1) / 2), math.floor((img.shape[COLS_AXIS] + 1 ) / 2)]
	# Step 2 - Estimate dimensions of final image after discrete log-polar transformation
	# num of columns = log(radius)
	# num of rows = angle in radius (0, 2pi)
    maxDiff = np.maximum(centerTrans, np.asarray(img.shape) - centerTrans)
    maxDistance = ((maxDiff[0] ** 2 + maxDiff[1] ** 2 ) ** 0.5)
    dimsLogPolar = [0,0]
    dimsLogPolar[COLS_AXIS] = img.shape[COLS_AXIS]
    dimsLogPolar[ROWS_AXIS] = img.shape[ROWS_AXIS]
    # Step 2.1 - Estimate log base
    logBase = math.exp(math.log(maxDistance) / dimsLogPolar[COLS_AXIS])
    # Step 3 - Calculate step for angle in log polar coordinates
    angleStep = ( 1.0 * math.pi ) / dimsLogPolar[ROWS_AXIS]
    return (centerTrans, angleStep, logBase)

# converts image to its log polar representation
# returns the log polar representation and log base
def convertToLogPolar(img, centerTrans, angleStep, logBase, mode = "nearest"):
    if mode == "nearest":
        # Step 1 - Initialize transformed image
        transformedImage = np.zeros(img.shape, dtype = img.dtype)
        # Step 2 - Apply reverse log polar transformation
        for radius in range(img.shape[COLS_AXIS]): # start with radius, because calculating exponential power is time consuming
            actRadius = logBase ** radius
            for angle in range(img.shape[ROWS_AXIS]):
                anglePi =  angle * angleStep
                # calculate euclidian coordinates (source: https://en.wikipedia.org/wiki/Log-polar_coordinates)
                row = int(centerTrans[ROWS_AXIS] + actRadius * math.sin(anglePi))
                col = int(centerTrans[COLS_AXIS] + actRadius * math.cos(anglePi))
                # copy pixel from the location to log polar image
                if 0 <= row < img.shape[ROWS_AXIS] and 0 <= col < img.shape[COLS_AXIS]:
                    transformedImage[angle, radius] = img[row, col]

        return transformedImage
    else:
        # print("Base: " + str(logBase))
        # create matrix with angles
        anglesMap = np.zeros(img.shape, dtype=np.float64)
        # each column has 0 in its first row and -pi in its last row
        anglesVector = -np.linspace(0, np.pi, img.shape[0], endpoint=False)
        # initialize it by columns using the same vector
        anglesMap.T[:] = anglesVector
        # create matrix with radii
        radiusMap = np.zeros(img.shape, dtype=np.float64)
        # each line contains a vector with numbers from  in (0, cols) to power logBase
        radiusVector = np.power(logBase, np.arange(img.shape[1], dtype=np.float64)) - 1.0
        # initialize it by rows using the same vector
        radiusMap[:] = radiusVector
        # calculate x coordinates (source: https://en.wikipedia.org/wiki/Log-polar_coordinates)
        x = radiusMap * np.sin(anglesMap) + centerTrans[1]
        # calculate y coordinates (source: https://en.wikipedia.org/wiki/Log-polar_coordinates)
        y = radiusMap * np.cos(anglesMap) + centerTrans[0]
        # initialize final image
        outputImg = np.zeros(img.shape)
        # use spline interpolation to map pixels from original image to calculated coordinates
        ndii.map_coordinates(img, [x, y], output=outputImg)
        return outputImg

#https://github.com/polakluk/fourier-mellin/blob/master/script.py
def calculateFourierMellin(img) :
    # Step 1 - Apply FFT on both images and get their magnitude spectrums
    imgFft, imgFftShifted = calculateFft(img)  # FFT of the image
    imgMags = cv.magnitude(imgFftShifted[:, :, RE_IDX], imgFftShifted[:, :, IM_IDX])


    cv.imshow("magnitude", histogramOperations.equalizeHistogram(np.uint8(util.normalize(imgMags, 1))))
    cv.waitKey(0)


    # Step 2 - Apply highpass filter on their magnitude spectrums
    highPassFilter = prepareHighPassFilter(imgMags)
    imgMagsFilter = imgMags * highPassFilter

    cv.imshow("filter", histogramOperations.equalizeHistogram(np.uint8(util.normalize(imgMagsFilter, 1))))
    cv.waitKey(0)

    # Step 3 - Convert magnitudes both images to log-polar coordinates
    # Step 3.1 - Precompute parameters (both images have the same dimensions)
    centerTrans, angleStep, logBase = computeLogPolarParameters(imgMagsFilter)
    imgLogPolar = convertToLogPolar(imgMagsFilter, centerTrans, angleStep, logBase, polarMode)

    cv.imshow("log polar", histogramOperations.equalizeHistogram(np.uint8(util.normalize(imgLogPolar, 1))))
    cv.waitKey(0)

    # Step 3.1 - Apply FFT on magnitude spectrums in log polar coordinates (in this case, not using FFT shift as it leads to computing [180-angle] results)
    imgLogPolarComplex = cv.dft(np.float32(imgLogPolar), flags=cv.DFT_COMPLEX_OUTPUT)
    imgLogPolarComplexMags = cv.magnitude(imgLogPolarComplex[:, :, RE_IDX], imgLogPolarComplex[:, :, IM_IDX])

    cv.imshow("complex mags", histogramOperations.equalizeHistogram(np.uint8(util.normalize(imgLogPolarComplexMags, 255))))
    cv.waitKey(0)

    return imgLogPolarComplexMags

def eliminateNoiseOnPattern(img, mask, window = 3) :
    img_float32 = np.float32(img)

    height, width = img.shape
    dftImage = np.zeros((height, width, 2 * (2 * window) * (2 * window)))
    rep = cv.copyMakeBorder(img_float32, window, window, window, window, cv.BORDER_REFLECT101)
    noiseFM = np.zeros(((2 * window), (2 * window), 2))
    for x in range(width) :
        xCoord = x + window
        for y in range(height) :
            yCoord = y + window
            currentPatch = rep[yCoord - window:yCoord + window, xCoord - window:xCoord + window]

            dft = cv.dft(currentPatch, flags=cv.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            dftImage[y, x] = (dft_shift.reshape((-1, 1))).transpose()
            if mask[y, x] == 1 :
                noiseFM += dft_shift

    noiseFM /= np.sum(mask)

    res = np.zeros(img.shape)
    for x in range(width) :
        for y in range(height) :
            dft_shift = dftImage[y, x].reshape((2 * window, 2 * window, 2))
            dft_shift -= noiseFM

            rows, cols = img.shape
            crow, ccol = rows / 2, cols / 2  # center

            # create a mask first, center square is 1, remaining all zeros
            # mask = np.zeros((rows, cols, 2), np.uint8)
            # mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
            #
            # # apply mask and inverse DFT
            # fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(dft_shift)
            img_back = cv.idft(f_ishift)
            img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            res[y, x] = img_back[window, window]

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(res, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()
    return res


def eliminateNoise(img, xNoise, yNoise, windowSize = 6, otherScales = True) :
    noiseHeight = windowSize
    noiseWidth = windowSize
    if noiseWidth % 2 == 1 :
        noiseWidth =  noiseWidth - 1
    if noiseHeight % 2 == 1 :
        noiseHeight = noiseHeight - 1
    noiseFM = calculateFourierMellin(img[yNoise:yNoise + noiseHeight, xNoise:xNoise + noiseWidth])
    depth = noiseWidth * noiseHeight
    fmWidth = noiseWidth
    fmHeight = noiseHeight
    noiseWidth = noiseWidth / 2
    noiseHeight = noiseHeight / 2
    noiseMean = np.mean(noiseFM)
    noiseFM = noiseFM - noiseMean
    height, width = img.shape
    result = np.zeros((height, width), np.float32)

    rep = cv.copyMakeBorder(img, noiseHeight, noiseHeight, noiseWidth, noiseWidth, cv.BORDER_REFLECT101)
    for x in range(width):
        xInd = x + noiseWidth
        for y in range(height):
            yInd = y + noiseHeight
            currentPatch = rep[yInd-noiseHeight:yInd+noiseHeight, xInd-noiseWidth:xInd+noiseWidth]
            imgFM = calculateFourierMellin(currentPatch)
            imgMean = np.mean(imgFM)
            imgFM = imgFM - imgMean
            corr = correlation(imgFM, noiseFM)
            result[y, x] = corr

        print x

    result2X = np.zeros((height, width))
    result3X = np.zeros((height, width))
    if otherScales :
        result2X = eliminateNoise(img, xNoise, yNoise, 2 * windowSize, False)
        result3X= eliminateNoise(img, xNoise, yNoise, 3 * windowSize, False)
        result = result + result2X + result3X
        result = result / 3
    return result

def correlation(img, noise) :
    img = img / 1000
    noise = noise / 1000
    numerator = np.multiply(img, noise)
    numerator = np.sum(numerator)
    imgPow = np.multiply(img, img.copy())
    noisePow = np.multiply(noise, noise.copy())
    denominator = np.multiply(imgPow, noisePow)
    denominator = np.sum(denominator)
    denominator = math.sqrt(denominator)
    corr = numerator / denominator
    if denominator == 0 or corr == np.nan:
        corr = 0
    return corr

def threeLayeredLearning(images, masks) :
    descriptors = []
    # i = (len(images)) - 1
    for i in range(len(images)) :
    # while i >= 0 :
        print("first Layer")
        #Thehistogram f!i of the original pattern sets of interest
        #of eachtrainingimage xi, andthethresholdparameter n to
        #determine the proportions of dominant patterns selected from
        #each training image.
        # descriptor = getDominantDescriptors(images[i], masks[i])
        name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/' + str(i) + '_FM.txt'
        # i = i - 1
        # np.savetxt(name, descriptor, delimiter=',')

        descriptor = np.loadtxt(name, delimiter=',')
        descriptors.append(descriptor)
    #Dominantpatternsets J1, J2,y, Jnj of nj images belonging to class j obtained from Algorithm 1.
    print("second Layer")
    descriptors = getDiscriminativeFeatures(descriptors)
    name = 'C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/training/discriminative_FM.txt'
    np.savetxt(name, descriptors, delimiter=',')
    #Thediscriminativedominantpatternset JCj for each class j  obtained from Algorithm 2.
    #since we only have one class we dont need this step
    # getGlobalFeatures()
    return descriptors

def getDominantDescriptors(img, mask, windowWidth = 5, windowHeight = 5, th = 1.4) :
    rep = cv.copyMakeBorder(img, windowHeight, windowHeight, windowWidth, windowWidth, cv.BORDER_REFLECT101)
    height, width = img.shape

    fmDescriptors = []
    for y in range(height) :
        yInd = y + windowHeight
        for x in range(width) :
            if mask[y, x] == 0 :
                continue
            xInd = x + windowWidth
            currentPatch = rep[yInd - windowHeight:yInd + windowHeight, xInd - windowWidth:xInd + windowWidth]
            imgFM = calculateFourierMellin(currentPatch)
            imgMean = np.mean(imgFM)
            #this line might be unnecessary when using other comparison methods
            imgFM = imgFM - imgMean
            imgFM = imgFM.flatten()
            fmDescriptors.append(imgFM)

    similarDescriptors = {}
    count = 0
    print (len(fmDescriptors))
    for currentPatch in fmDescriptors:
        similarFound = False
        print count
        if count > 7000 :
            break
        print count
        count = count + 1
        for i in similarDescriptors:
            corr = correlation(currentPatch, np.asarray(i))
            if corr > th:
                similarDescriptors[i] = similarDescriptors[i] + 1
                similarFound = True
                break
        if not similarFound :
            tup = tuple(currentPatch)
            similarDescriptors[tup] = 1

    sortedDescriptors = sorted(similarDescriptors.items(), key=operator.itemgetter(1))
    dominantDescriptors = []
    for currentPatch in reversed(sortedDescriptors) :
        if sortedDescriptors[1] < 10 :
            break
        dominantDescriptors.append(np.asarray(currentPatch[0]))

    return dominantDescriptors

def getDiscriminativeFeatures(features, th = 1.4) :
    discriminativeFeatures = features[0]
    i = 1
    while i < len(features) :
        print{"..."}
        patches = features[i]
        intersection = []
        for feature in discriminativeFeatures:
            print("sorting")
            for patch in patches:
                corr = correlation(feature, patch)
                # print corr
                if corr > th :
                    intersection.append(patch)
                    break
        discriminativeFeatures = intersection
        i += 1

    return discriminativeFeatures