import cv2 as cv
def normalize(img, maxVal):
    min, max, minLoc, maxLoc = cv.minMaxLoc(img)
    img = (img - min) / (max - min)
    img = img * maxVal;
    return img