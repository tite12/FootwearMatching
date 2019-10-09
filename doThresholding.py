import cv2 as cv
import numpy

def blur(img) :
    return cv.medianBlur(img, 3);

def adaptiveThreshold(img) :
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

def otsuThreshold(img) :
    retval, img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return img

def fuyyzEnhancement(img) :
    orig = img.copy()
    min, max, minLoc, maxLoc = cv.minMaxLoc(img)
    height, width = img.shape
    for x in range(width-1):
        for y in range(height-1):
            currentVal = img[y, x]
            currentVal = (currentVal - min)/(max-min)

            if currentVal < 0.5 :
                currentVal = 2 * pow(currentVal, 2)
            else:
                currentVal = 1 - (2 * pow((1-currentVal), 2))
            img[y, x] = currentVal

    cv.imshow("img", img)
    cv.imshow("orig", orig)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img