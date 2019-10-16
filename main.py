import numpy as np
import cv2 as cv


import doThresholding
import histogramOperations
import filters
import util
import enhancement

img = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/hard/00232.jpg', 0)


bgrImg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
epf1 = filters.epf(bgrImg, 3, 10)
epf2 = filters.epf(bgrImg, 3, 120)
epf3 = filters.epf(bgrImg, 3, 250)

cv.imshow("orig", img)

cv.imshow("epf1", epf1)
cv.imshow("epf2", epf2)
cv.imshow("epf3", epf3)
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