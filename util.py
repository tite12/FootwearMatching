import cv2 as cv
from random import seed
from random import randint

def normalize(img, maxVal):
    min, max, minLoc, maxLoc = cv.minMaxLoc(img)
    img = (img - min) / (max - min)
    img = img * maxVal;
    return img

def generateColor(seedVal = 0, min = 0, max = 255) :
    seed(seedVal)
    return [randint(min, max), randint(min, max), randint(min, max)]