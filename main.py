# Dataset generálás

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random as r
# Create a black image
size=128
for i in range(500):
    #circleNum=r.randint(1,3)
    circleNum=1
    print(i)
    xImg = np.zeros((128, 128, 3), np.uint8)
    yImg = np.zeros((128, 128, 3), np.uint8)
    for j in range(circleNum):
        
        xX = np.random.randint(10, 118)
        xY = np.random.randint(10, 118)
        yX = 128-xX
        yY = 128-xY
        radius = 5
        # Draw a diagonal blue line with thickness of 5 px
        #cv.rectangle(img, (upX, upY), (bottomX, bottomY), (0, 255, 0), 3)

        cv.circle(xImg, (xX, xY), radius, (255, 0, 0), -1)
        cv.circle(yImg, (yX, yY), radius, (255, 0, 0), -1)



    plt.imsave("test/"+str(i)+"_groundTruth.png", xImg)
    plt.imsave("test/"+str(i)+"_noised.png", yImg)
    # with open('labels.txt', 'a') as f:
    #    f.write(str(i)+"_noised.png," +str(i)+"_groundTruth.png"+ '\n')
