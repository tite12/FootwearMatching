import numpy as np
import cv2 as cv


import doThresholding
import histogramOperations
import filters
import util
import enhancement

img = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/hard/00232.jpg', 0)

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
cv.imshow("adaptive", adaptive)
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