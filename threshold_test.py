import time
import cv2 as cv
import numpy as np
from board_func import whiteboard
from blend_modes import multiply

import tkinter as tk
from PIL import Image, ImageTk

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype('int').tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)

import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Drag and Drop")
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

hold = 0

vid = cv.VideoCapture("http://192.168.147.55:4747/video")

last_good_corners = np.zeros((4, 1, 2))
filter_toggle = 0

while(True):
    ret, img = vid.read()

    scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    imgorg = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    cv.imshow('ORIGINAL',imgorg)


    #grayscale
    imgrs = cv.cvtColor(imgorg, cv.COLOR_BGR2GRAY)
    img = imgrs.copy()

    ret,img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #cv.imshow('thresh-output',img)

    siz = 5
    kernel = np.ones((siz,siz),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    #cv.imshow('pre-output',img)

    siz = 10
    kernel = np.ones((siz,siz),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    img = cv.erode(img,kernel,iterations = 1)
    #cv.imshow('post-output',img)

    over_dil = cv.dilate(img,kernel,iterations = 3)

    contours, _ = cv.findContours(over_dil, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    x, y, w, h = cv.boundingRect(max(contours, key=cv.contourArea))
    #cv.rectangle(imgrs, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #new_img = img[y:y+h,x:x+w].copy()

    # Define the selected index range
    slice_row = slice(y, y+h)
    slice_col = slice(x, x+w)
    # Set all elements outside the selected index range to zero
    img[:slice_row.start] = 0
    img[slice_row.stop:] = 0
    img[:, :slice_col.start] = 0
    img[:, slice_col.stop:] = 0

    canny = cv.Canny(img, 0, 200)
    canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    edg_contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    page = sorted(edg_contours, key=cv.contourArea, reverse=True)[:5]

    # Loop over the contours.

    fckd = 0

    for c in edg_contours:
        # Approximate the contour.
        epsilon = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points
        if len(corners) == 4:
            temp_corners = corners.copy()
            break
        else:
            fckd = 1
    
    if (hold == 0):
        if (fckd == 0):
            pg_corners = temp_corners.copy()
        else:
            pg_corners = last_good_corners.copy()

    """ cv.drawContours(canny, c, -1, (50,50,50), 3)
    cv.drawContours(canny, corners, -1, (50,50,50), 10)
    corners = sorted(np.concatenate(corners).tolist())
    for index, c in enumerate(corners):
    character = chr(65 + index)
    cv.putText(canny, character, tuple(c), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 1, cv.LINE_AA) """
    
    #if len(corners) == 4:
    # Sorting the corners and converting them to desired shape.
    sort_corners = sorted(np.concatenate(pg_corners).tolist())
    # For 4 corner points being detected.
    ord_corners = order_points(sort_corners)
    destination_corners = find_dest(ord_corners)
    # Getting the homography.
    M = cv.getPerspectiveTransform(np.float32(ord_corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv.warpPerspective(imgorg, M, (destination_corners[2][0], destination_corners[2][1]),flags=cv.INTER_LINEAR)
    #else:
    #    print(len(corners))

    hold+=1
    if (hold == 20): hold = 0

    #TODO: Scale the page here----------------###################################
    scale_percent = 250 # percent of original size
    width = int(final.shape[1] * scale_percent / 100)
    height = int(final.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    sc_final = cv.resize(final, dim, interpolation = cv.INTER_LINEAR)
    sc_final = cv.flip(sc_final,-1)

    """text = str(hold)
    font = cv.FONT_HERSHEY_PLAIN
    font_scale = 1
    color = (255, 0, 0)  # BGR format
    thickness = 1
    position = (50, 50)
    cv.putText(sc_final, text, position, font, font_scale, color, thickness)"""

    #cv.imshow('extract-output',sc_final)

    if (filter_toggle == 1):
        postproc_out = whiteboard(sc_final)
    else:
        postproc_out = sc_final
    # Get the image shape
    #height, width = postproc_out.shape[:2][::-1]
    #print(postproc_out.shape[:2][::-1][0])
    
    # Define the crop region
    crop_top = 10
    crop_bottom = postproc_out.shape[:2][::-1][1] - 10
    crop_left = 10
    crop_right = postproc_out.shape[:2][::-1][0] - 10
    
    # Crop the image
    board_crop = postproc_out[crop_top:crop_bottom, crop_left:crop_right]
    cv.imshow('whiteboard-output',board_crop)

    heart = cv.imread("D:\\Python\\Webcam\\heart.png")
    scale_percent = 70 # percent of original size
    width = int(heart.shape[1] * scale_percent / 100)
    height = int(heart.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    heart = cv.resize(heart, dim, interpolation = cv.INTER_LINEAR)
    #heart = cv.cvtColor(heart,cv.COLOR_RGB2BGR)
    canvas = np.ones_like(board_crop)

    # Get the dimensions of the large and small images
    lh, lw, _ = canvas.shape
    sh, sw, _ = heart.shape
    
    # Define the top-left corner of the small image in the large image
    x_offset = 50
    y_offset = 50
    
    # Define the bottom-right corner of the small image in the large image
    x_end = x_offset + sw
    y_end = y_offset + sh
    
    cv.rectangle(canvas, (0, 0), (lw, lh), (255, 255, 255), -1)
    
    # Copy the pixel values of the small image into the large image
    canvas[y_offset:y_end, x_offset:x_end] = heart
    a = board_crop.astype(float)
    b = canvas.astype(float)

    a = np.dstack((a, np.ones((a.shape[0], a.shape[1]), dtype=float)))
    b = np.dstack((b, np.ones((b.shape[0], b.shape[1]), dtype=float)))
    a *= 1
    b *= 1

    ab = multiply(b,a,0.8)
    ab = ab.astype(np.uint8)

    heart_image_id = canvas.create_image(50, 50, image=ab, anchor=tk.NW)
    heart_photo = ImageTk.PhotoImage(ab)

    canvas.itemconfig(heart_image_id, image=heart_photo)
    
    root.update()

    #cv.imshow('heart-png',ab)
    

    # exit on q key
    key = cv.waitKey(5)
    if key == ord('s'):
        last_good_corners = pg_corners.copy()

    if key == ord('q'):
        break

    if key == ord('f'):
        filter_toggle ^= 1

    # ret: a boolean value that indicates whether the frame was successfully read or not.
    if ret == False: break

vid.release()
cv.destroyAllWindows()