import numpy as np
import cv2
from matplotlib import pyplot as plt

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)
while True:
    __, img1 = cam1.read()
    __, img2 = cam2.read()
    img1_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # cv2.imshow('1', img1)
    # cv2.imshow('2', img2)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img1_1, img2_2)
    plt.imshow(disparity, 'gray')
    plt.show()
    k = cv2.waitKey(30) & 0xff
    if k == 27: break

cam1.release()
cam2.release()