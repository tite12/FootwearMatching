import numpy as np
import cv2 as cv


import doThresholding
import histogramOperations
import filters
import util

img = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/hard/00233.jpg', 0)

img = np.float32(img)
img = img * 1.0/255

# img = doThresholding.calculateSkeleton(img)
wienerFilteredImg = filters.wiener(img.copy())
invertedWienerFiltered = 1.0-wienerFilteredImg
wienerImg = img - invertedWienerFiltered

wienerImg = util.normalize(wienerImg, 1.0)
wienerImg = np.float32(wienerImg)

fuzzyImg = doThresholding.fuzzyEnhancement(wienerImg.copy())
adaptiveImg = doThresholding.adaptiveEnhancement(fuzzyImg.copy())

adaptiveImg = util.normalize(adaptiveImg, 255)
adaptiveImg = adaptiveImg.astype(np.uint8)

# smqtImg = doThresholding.fastSMQT(adaptiveImg.copy())

# faDiff = fuzzyImg - adaptiveImg
# asdiff = adaptiveImg - smqtImg
# asdiff = util.normalize(asdiff, 255)
# asdiff = asdiff.astype(np.uint8)

skeletonInvImg = doThresholding.calculateSkeleton(255 - adaptiveImg)
skeletonImg = doThresholding.calculateSkeleton(adaptiveImg.copy())
otsuThImg = doThresholding.otsuThreshold(adaptiveImg.copy())
adaptiveThImg = doThresholding.adaptiveThreshold(adaptiveImg.copy())

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

cv.imshow("otsu th img", otsuThImg)
cv.imshow("adaptive th img", adaptiveThImg)
cv.imshow("skeleton img", skeletonImg)
cv.imshow("skeleton Inv img", skeletonInvImg)

# cv.imshow("diff otsu th img", diffSkeletonInvImg)
# cv.imshow("diff adaptive th img", diffAdaptiveThImg)
# cv.imshow("diff skeleton img", diffSkeletonImg)
# cv.imshow("diff skeleton Inv img", diffSkeletonInvImg)

cv.waitKey(0)
cv.destroyAllWindows()

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
kernel = np.ones((2,2),np.uint8)
# img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)


# cv.imshow("img",img)
# cv.imshow("orig",orig)
# cv.waitKey(0)
# cv.destroyAllWindows()