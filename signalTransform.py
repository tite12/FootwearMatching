import numpy as np
import cv2 as cv
import math
import scipy.ndimage.interpolation as ndii
import util
import histogramOperations

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

    # Step 2 - Apply highpass filter on their magnitude spectrums
    highPassFilter = prepareHighPassFilter(imgMags)
    imgMagsFilter = imgMags * highPassFilter

    # Step 3 - Convert magnitudes both images to log-polar coordinates
    # Step 3.1 - Precompute parameters (both images have the same dimensions)
    centerTrans, angleStep, logBase = computeLogPolarParameters(imgMagsFilter)
    imgLogPolar = convertToLogPolar(imgMagsFilter, centerTrans, angleStep, logBase, polarMode)

    # Step 3.1 - Apply FFT on magnitude spectrums in log polar coordinates (in this case, not using FFT shift as it leads to computing [180-angle] results)
    imgLogPolarComplex = cv.dft(np.float32(imgLogPolar), flags=cv.DFT_COMPLEX_OUTPUT)
    imgLogPolarComplexMags = cv.magnitude(imgLogPolarComplex[:, :, RE_IDX], imgLogPolarComplex[:, :, IM_IDX])

    return imgLogPolarComplexMags

def eliminateNoise(img, noise) :
    # img = calculateFourierMellin(img)
    # img = np.uint8(util.normalize(img, 255))
    # img = histogramOperations.equalizeHistogram(img)
    # cv.imshow("F-M", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    noiseHeight, noiseWidth = noise.shape
    if noiseWidth % 2 == 1 :
        noiseWidth =  noiseWidth - 1
    if noiseHeight % 2 == 1 :
        noiseHeight = noiseHeight - 1
    noiseFM = calculateFourierMellin(noise[0:noiseHeight, 0:noiseWidth])
    noiseWidth = noiseWidth / 2
    noiseHeight = noiseHeight / 2
    noiseMean = np.mean(noiseFM)
    noiseFM = noiseFM - noiseMean
    height, width = img.shape
    result = np.zeros((height, width), np.float32)
    # cv.imshow("noise", noiseFM)
    for x in range(width):
        if x - noiseWidth < 0 or x + noiseWidth > width:
            continue
        for y in range(height):
            if y - noiseHeight < 0 or y + noiseHeight > height :
                continue
            currentPatch = img[y-noiseHeight:y+noiseHeight, x-noiseWidth:x+noiseWidth]
            imgFM = calculateFourierMellin(currentPatch)
            imgMean = np.mean(imgFM)
            imgFM = imgFM - imgMean
            # cv.imshow('img', imgFM)
            # cv.waitKey(0)
            corr = correlation(imgFM, noiseFM)
            # print(corr)
            result[y, x] = corr

        print x
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
    return corr