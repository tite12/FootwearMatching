import cv2 as cv
import numpy as np
import math
from scipy import ndimage
import pywt
import util

def adaptiveThreshold(img) :
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

def otsuThreshold(img) :
    retval, img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return img

def niblackThreshold(img, binMethod = 0, blockSize = 5, k = 0.5) :
    binarization = cv.ximgproc.BINARIZATION_NIBLACK
    if binMethod == 1 :
        binarization = cv.ximgproc.BINARIZATION_SAUVOLA
    elif binMethod == 2 :
        binarization = cv.ximgproc.BINARIZATION_WOLF
    elif binMethod == 3 :
        binarization = cv.ximgproc.BINARIZATION_NICK
    dst = np.zeros(img.shape)
    # dst = cv.ximgproc.niBlackThreshold(img, 1.0, cv.THRESH_BINARY, blockSize, k)
    dst = cv.ximgproc.niBlackThreshold(img, 1.0, cv.THRESH_BINARY, blockSize, k, dst, binarization)
    return dst

def calculateSkeleton(img) :
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv.threshold(img, 127, 255, 0)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def thinning(img) :
    dst = cv.ximgproc.thinning(img)
    return dst

def canny(img, noiseImg, sigma = 0.33) :
    med = np.median(img * noiseImg)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))

    edges = cv.Canny(img, lower, upper) * noiseImg
    return edges

def holisticallyNestedEdgeDetection(img, noiseImg , register = False) :
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    if register :
        # ! [Register]
        cv.dnn_registerLayer('Crop', CropLayer)
        # ! [Register]


    # Load the model.
    net = cv.dnn.readNetFromCaffe("C:/Users/rebeb/Documents/TU_Wien/Dipl/project/caffe/deploy.prototxt",
                                  "C:/Users/rebeb/Documents/TU_Wien/Dipl/project/caffe/hed_pretrained_bsds.caffemodel")

    height, width, _ = img.shape
    inp = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(width, height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)

    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (img.shape[1], img.shape[0]))

    out = out * noiseImg
    return out

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]