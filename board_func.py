import cv2 as cv
import numpy as np


def get_red_blob(img):
    img = cv.bilateralFilter(img, 9, 75, 75)
    img = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define range of red color in HSV
    lower_red = np.array([0, 20, 20])
    upper_red = np.array([30, 255, 255])
    lower_red2 = np.array([160, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    
    # Create a mask for the red color range
    mask1 = cv.inRange(hsv, lower_red, upper_red)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)
    kernel = np.ones((5,5),np.uint8)
    mask = cv.dilate(mask,kernel,iterations = 3)
    mask = mask.astype(np.uint8)

    return mask



def whiteboard(imgorg):

    img = imgorg.copy()

    red_reg = get_red_blob(img)

    #kernel = np.ones((5,5),np.uint8)
    #img = cv.erode(img,kernel,iterations = 1)

    #grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img = clahe.apply(img)
    #img = cv.bilateralFilter(img, 9, 75, 75)

    img = cv.bitwise_not(img)

    #blur image before thresholding to fill any gaps
    #kernel = np.ones((5,5),np.uint8)
    #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations= 3)

    #threshold
    T_, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    img = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
    img = cv.convertScaleAbs(img, alpha=3, beta=0)

    kernel = np.ones((2,2),np.uint8)
    img = cv.dilate(img,kernel,iterations = 1)

    #img = cv.medianBlur(img,7)

    kernel = np.ones((3,3),np.uint8)
    img = cv.erode(img,kernel,iterations = 1)

    #canny = cv.Canny(img, 50, 200)

    #img = cv.bitwise_not(img)

    # Create a pure red array in the same size as the image
    red_pure = np.ones_like(imgorg) * (255,255,0)
    red_reg = cv.cvtColor(red_reg, cv.COLOR_GRAY2BGR)

    # Create a black array in the same size as the image
    black_pure = np.ones_like(imgorg)
    black_reg = cv.bitwise_not(red_reg)

    img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    #red_blob = cv.bitwise_and(red_pure,red_pure,mask=red_reg).astype(np.uint8)
    red_blob = np.bitwise_and(red_pure,red_reg)
    red_let = np.bitwise_and(red_blob,img).astype(np.uint8)
    #cv.imshow('red_let',red_let)

    #black_blob = np.bitwise_and(black_reg,black_pure)
    black_let = np.bitwise_and(np.bitwise_not(red_reg),img).astype(np.uint8)
    #cv.imshow('black_let',black_let)

    final = cv.bitwise_or(black_let,red_let)
    final = cv.bitwise_not(final)

    return final