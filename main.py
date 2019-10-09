import numpy as np
import cv2 as cv

import doThresholding

img = cv.imread('C:/Users/rebeb/Documents/TU_Wien/Dipl/FID-300/FID-300/FID-300/test_images/easy/00204.jpg', 0)
img = np.float32(img)
img = img * 1.0/255
orig = img.copy();
doThresholding.fuyyzEnhancement(img)
# for x in range(3):
#     img = doThresholding.blur(img)
#     img = doThresholding.otsuThreshold(img)
#     cv.imshow("img", img)
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