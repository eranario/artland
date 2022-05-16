# extract region of interest from depth image using thresholding with RGB image

import cv2 as cv
import numpy as np
from PIL import Image

# load the image
img = cv.imread('D:/artland/data/2022_4_27_yb_ambient/rgb_2022_4_27_1651120441.png')
depth = cv.imread('D:/artland/data/2022_4_27_yb_ambient/depth_2022_4_27_1651120441.png')

# define the list of boundaries in BGR
boundaries = [
    ([50, 0, 0], [220, 88, 50]) # blue
]

# loop over boundaries
for (lower,upper) in boundaries:
    # create numpy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv.inRange(img, lower, upper)
    # output = cv.bitwise_and(img, img, mask=mask) 

    # apply mask to depth image
    depth_masked = cv.bitwise_and(depth, depth, mask=mask)
    print(depth_masked.shape)
    cv.imshow("Mask Applied to Image", depth_masked)
    cv.waitKey(0)

    # # show image
    # cv.imshow("image", np.hstack([img,output]))
    # cv.waitKey(0)