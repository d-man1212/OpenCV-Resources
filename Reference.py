import cv2
import numpy as np
from copy import deepcopy
print("Package Imported")
"""SUGGESTIONS: When you initialized using \ separator, your code will work only for Windows.

imgTarget = cv.imread('photos\TargetImage.jpg') #bu resmimiz
myVid = cv.VideoCapture('photos\video.mp4')
If you use os.path's sep, it will work on all OS.

from os.path import sep

imgTarget = cv.imread('photos' + sep + 'TargetImage.jpg') #bu resmimiz
myVid = cv.VideoCapture('photos' + sep + 'video.mp4') """

def imgcapture():
    img = cv2.imread("Images/chad.png")                                     # Read image
    cv2.imshow("GIGACHAD", img)                                             # Display image, arguments: Name of Window, img file
    print(img.shape)                                                        # Displays size in pixels in height, width, no of channels (y, x, bgr) format
    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

    imgResize = cv2.resize(img, (900,400))                                  # Resize image in specified width-height format
    cv2.imshow("WIDECHAD", imgResize)                                       # Display image, arguments: Name of Window, img file
    print(imgResize.shape)                                                  # Displays size in pixels in height, width, no of channels (y, x, bgr) format
    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

    Croppedimg = img[0:img.shape[0],0:img.shape[1]//2]                      # Format: start and stop indices - Height, Width of img
    cv2.imshow("CROPPEDCHAD", Croppedimg)                                   # Display image, arguments: Name of Window, img file
    print(Croppedimg.shape)                                                 # Displays size in pixels in height, width, no of channels (y, x, bgr) format
    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def imgCreate():
    #img = np.zeros((512,256))                                              # Create grayscale image of 512*256 res (Height, Width)
    #cv2.imshow("Grayscaled Image", img)                                    # Display image, arguments: Name of Window, img file
    #print(img.shape)                                                       # Displays size in pixels in height, width, no of channels (y, x, bgr) format
    #cv2.waitKey(0)                                                         # Infinite delay else image closes instantly or show for specified ms

    bgrimg = np.zeros((512,512,3),np.uint8)                                 # Create bgrband image of 512*512 res (Height, Width)
    #whiteimg = deepcopy(bgrimg)                                            # All elements are black - 0's, so in order to change color of parts elements should be changed to whatever bgr value
    #whiteimg[:] = 255,255,255                                              # White bgr
    #cv2.line(whiteimg, (0,0), (512,512), (0, 0, 0), 5)                     # Format: image file, start point, end point, color of line, thickness (point format:width, height)

    bgrimg[0:512,0:512//3] = 255, 0, 0                                      # Format: start and stop indices - Height, Width of img
    bgrimg[0:512,512//3:2*512//3] = 0, 255, 0                               # Format: start and stop indices - Height, Width of img
    bgrimg[0:512,2*512//3:512] = 0, 0, 255                                  # Format: start and stop indices - Height, Width of img
    cv2.rectangle(bgrimg, (0,0), (511//3,511), (0, 0, 0), 5)                # Format: image file, start point, diagonal point, color of line, thickness (point format:width, height)
    cv2.rectangle(bgrimg, (0,0), (2*511//3,511), (0, 0, 0), 5)              # Format: image file, start point, diagonal point, color of line, thickness (point format:width, height)
    cv2.rectangle(bgrimg, (0,0), (511,511), (0, 0, 0), 5)                   # Format: image file, start point, diagonal point, color of line, thickness (point format:width, height)
    cv2.rectangle(bgrimg,(0,206),(512,306),(255, 255, 255),cv2.FILLED)      # Format: image file, start point, diagonal point, color of line, fill (point format:width, height)
    cv2.circle(bgrimg, (256,256), 50, (0, 0, 0), 5)                         # Format: image file, center, radius (point format:width, height)
    cv2.putText(bgrimg,"The Divided States",(10,500),
                cv2.FONT_HERSHEY_COMPLEX,1, (84,80,196),1)                  #Format: image file, origin, font, scale, color, thickness

    cv2.imshow("BGR Band Image", bgrimg)                                    # Display image, arguments: Name of Window, img file
    #cv2.imshow("White Image", whiteimg)                                    # Display image, arguments: Name of Window, img file
    print(bgrimg.shape)                                                     # Displays size in pixels in height, width, no of channels (y, x, bgr) format

    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def imgGrayScale():
    img = cv2.imread("Images/chad.png")                                     # Read image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # Convert to particular color space - grayscale (bgr format cv2)
    cv2.imshow("GraygaChad", imgGray)                                       # Display image, arguments: Name of Window, img file
    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def imgBlur():
    img = cv2.imread("Images/chad.png")                                     # Read image
    imgBlur = cv2.GaussianBlur(img, (7,7), 0)                               # Requires img, kernel size - amt of blur (has to be odd), sigma x = 0
    cv2.imshow("BluraChad", imgBlur)                                        # Display image, arguments: Name of Window, img file
    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def imgCanny():
    img = cv2.imread("Images/chad.png")                                     # Read image
    imgCanny = cv2.Canny(img, 50, 100)                                      # Requires img, threshold values
    cv2.imshow("CannyChad", imgCanny)                                       # Display image, arguments: Name of Window, img file

    kernel=np.ones((5,5),np.uint8)                                          # size of matrix,type of object 8 bit 0 - 255
    imgdilate = cv2.dilate(imgCanny, kernel, iterations = 5 )               # Requires img, kernel, thickness
    cv2.imshow("DilateChad", imgdilate)                                     # Display image, arguments: Name of Window, img file

    imgerode = cv2.erode(imgdilate, kernel, iterations = 1)                 # Requires img, kernel, thickness
    cv2.imshow("ErodeChad", imgerode)                                       # Display image, arguments: Name of Window, img file

    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def videocapture():
    vid = cv2.VideoCapture("Videos/GUSJOJO.mp4")                            # Read video
    while 1:                                                                # Reading each frame one by one
        success, frame = vid.read()                                         # Boolean for success or failure and image frame file
        cv2.imshow("Breaking Bell", frame)                                  # Display frame, arguments: Name of Window, img file
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break                                                           # Close after q is pressed

def webcamcapture():
    vid = cv2.VideoCapture(0)                                               # 0 - default webcam, 1 - next webcam

    vid.set(3, 640)                                                         # width as 640 px
    vid.set(4, 480)                                                         # height as 480 px
    vid.set(10, 100)                                                        # brightness as 100

    while 1:                                                                # Reading each frame one by one
        success, frame = vid.read()                                         # Boolean for success or failure and image frame file
        cv2.imshow("Hello DMan", frame)                                     # Display frame, arguments: Name of Window, img file
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break                                                           # Close after q is pressed

def cardread():
    img = cv2.imread("Images/cards.png")                                    # Read image

    pts1 = np.float32([[223,90], [435, 135], [160,382], [373,428]])         # See values of pixels at the bottom by opening img in MS Paint (A,B,D,C)
    width, height = 350, 500                                                # Width and heights of cards depicted
    pts2 = np.float32([[0, 0],[width, 0],[0, height],[width, height]])      # What the above points reference (A,B,D,C)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)                        # Transform the cut image into pts2
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))           # Format: img file, matrix, width and height required

    cv2.imshow("Image", img)                                                # Display frame, arguments: Name of Window, img file
    cv2.imshow("Output", imgOutput)                                         # Display frame, arguments: Name of Window, img file

    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def imgMulti():
    """
    img1 = cv2.imread("Images/chad.png")                                    # Read image
    img2 = cv2.imread("Images/cards.png")                                   # Read image
    imghorizontalstack = np.hstack((img1,img1))                             # Dimensions of stacked images must match and also the no of color channels
    imgverticalstack = np.vstack((img2,img2))                               # Dimensions of stacked images must match and also the no of color channels
    cv2.imshow("HSTACK", imghorizontalstack)                                # Display frame, arguments: Name of Window, img file
    cv2.imshow("VSTACK", imgverticalstack)                                  # Display frame, arguments: Name of Window, img file

    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms """

    img = cv2.imread("Images/chad.png")                                     # Read image
    img2 = cv2.imread("Images/cards.png")                                   # Read image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # Grayscale image
    imgStack = stackImages(0.7, ([img, imgGray, img2], [img, img, img]))    # Format: Scale, matrix of img example [img,img,img]
    cv2.imshow("ImageStack", imgStack)                                      # Display frame, arguments: Name of Window, img file
    cv2.waitKey(0)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass

def colordetect():
    cv2.namedWindow("TrackBars")                                            # Create a window
    cv2.resizeWindow("TrackBars",640,240)                                   # Resize window res
    
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179,empty)                # Format: Title, window, start, stop, function
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179,empty)              # Format: Title, window, start, stop, function
    cv2.createTrackbar("Saturation Min", "TrackBars", 0, 255,empty)         # Format: Title, window, start, stop, function
    cv2.createTrackbar("Saturation Max", "TrackBars", 255, 255,empty)       # Format: Title, window, start, stop, function
    cv2.createTrackbar("Value Min", "TrackBars", 0, 255,empty)              # Format: Title, window, start, stop, function
    cv2.createTrackbar("Value Max", "TrackBars", 255, 255,empty)            # Format: Title, window, start, stop, function

    while 1:
        img = cv2.imread("Images/chad.png")                                 # Read image
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)                        # Convert image into HSV Space

        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")                  # Get trackbar value of particular trackbar (trackbar, window)
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")                  # Get trackbar value of particular trackbar (trackbar, window)
        s_min = cv2.getTrackbarPos("Saturation Min", "TrackBars")           # Get trackbar value of particular trackbar (trackbar, window)
        s_max = cv2.getTrackbarPos("Saturation Max", "TrackBars")           # Get trackbar value of particular trackbar (trackbar, window)
        v_min = cv2.getTrackbarPos("Value Min", "TrackBars")                # Get trackbar value of particular trackbar (trackbar, window)
        v_max = cv2.getTrackbarPos("Value Max", "TrackBars")                # Get trackbar value of particular trackbar (trackbar, window)
        #print(h_min,h_max,s_min,s_max,v_min,v_max,sep=" ")

        lower = np.array([h_min,s_min,v_min])                               # numpy array of lower values
        upper = np.array([h_max,s_max,v_max])                               # numpy array of upper values


        mask = cv2.inRange(imgHSV,lower,upper)                              # masks filters out all the colors between the range and returns the image
        imgResult = cv2.bitwise_and(img, img, mask=mask)                    # Apply mask on image

        #cv2.imshow("GIGAChad", img)                                        # Display image, arguments: Name of Window, img file
        #cv2.imshow("HSVChad", imgHSV)                                      # Display image, arguments: Name of Window, img file
        #cv2.imshow("MASK", mask)                                           # Display image, arguments: Name of Window, img file
        #cv2.imshow("Result", imgResult)                                    # Display image, arguments: Name of Window, img file
        imgStack = stackImages(1, ([img, mask, imgResult]))                 # Format: Scale, matrix of img example [img,img,img]
        cv2.imshow("ImageStack", imgStack)                                  # Display frame, arguments: Name of Window, img file
        cv2.waitKey(1)                                                      # Infinite delay else image closes instantly or show for specified ms

def shapedetect():
    img = cv2.imread("Images/shapes.png")                                   # Read image
    imgContour = deepcopy(img)                                              # Duplicate Image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # Convert to particular color space - grayscale (bgr format cv2)
    imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)                           # Requires img, kernel size - amt of blur (has to be odd), sigma x = 0 (more blur)
    imgCanny = cv2.Canny(imgBlur, 100, 100)                                 # Requires img, threshold values
    getContours(imgCanny,imgContour)                                        # Draw all contours on imgContour
    blankimg = np.zeros_like(img)
    imgStack = stackImages(1, ([[img, imgGray, imgBlur],
                             [blankimg, imgCanny, imgContour]]))            # Format: Scale, matrix of img example [img,img,img]
    cv2.imshow("ImageStack", imgStack)                                      # Display frame, arguments: Name of Window, img file
    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)           # Find contours of image by specifying a method (retrieves extreme outer contours) and request all contours or compressed
    for cnt in contours:
        area = cv2.contourArea(cnt)                                         # Find area of each contour
        """if area>50:                                                        # Avoids noise
            cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3)             # Format: img, contour, contour index (-1) for all contours, color, thickness"""
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)               # Format: img, contour, contour index (-1) for all contours, color, thickness
        peri = cv2.arcLength(cnt, True)                                     # Perimeter / Arc length of each contour
        # print(peri)                                                       # Prints perimeter
        # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)                   # Used to approximate the contour with a polygonal curve, where the second argument 0.02 * peri specifies the maximum distance between the original contour and its approximation. cv2.approxPolyDP(curve, epsilon, closed)
        # print(len(approx))
        #objCor = len(approx)                                                # Gives no of sides present in each shape.
        # x, y, w, h = cv2.boundingRect(approx)                               # Draws a bounding rect on top of shape
        # if objCor == 3: objectType = "Tri"
        # elif objCor == 4:
        #     aspRatio = w / float(h)
        #    if aspRatio > 0.98 and aspRatio < 1.03: objectType = "Square"
        #    else: objectType = "Rectangle"
        # elif objCor > 4: objectType = "Circles"
        # else: objectType = "None"
        # cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(imgContour, objectType,
        #            (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
        #           (0, 0, 0), 2)

def shapedetectcustom1Tracker(path):
    img = cv2.imread(path)                                                  # Read image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # Convert to particular color space - grayscale (bgr format cv2)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)                          # Requires img, kernel size - amt of blur (has to be odd), sigma x = 0 (more blur)
    cv2.namedWindow("TrackBars")                                            # Create a window
    cv2.resizeWindow("TrackBars", 640, 240)                                 # Resize window res
    cv2.createTrackbar("Threshold", "TrackBars", 0, 500, empty)             # Format: Title, window, start, stop, function
    while 1:
        imgContour = deepcopy(img)                                          # Duplicate Image
        t = cv2.getTrackbarPos("Threshold", "TrackBars")                    # Get trackbar value of particular trackbar (trackbar, window)
        imgCanny = cv2.Canny(imgBlur, t, t)                                 # Requires img, threshold values
        getContours(imgCanny, imgContour)                                   # Draw all contours on imgContour
        imgStack = stackImages(1, ([img, imgCanny, imgContour]))            # Format: Scale, matrix of img example [img,img,img]
        cv2.imshow("ImageStack", imgStack)                                  # Display frame, arguments: Name of Window, img file
        cv2.waitKey(1)                                                      # Infinite delay else image closes instantly or show for specified ms

def shapedetectcustom2Tracker(path):
    img = cv2.imread(path)                                                  # Read image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # Convert to particular color space - grayscale (bgr format cv2)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)                          # Requires img, kernel size - amt of blur (has to be odd), sigma x = 0 (more blur)
    cv2.namedWindow("TrackBars")                                            # Create a window
    cv2.resizeWindow("TrackBars", 640, 240)                                 # Resize window res
    cv2.createTrackbar("Threshold 1", "TrackBars", 0, 500, empty)           # Format: Title, window, start, stop, function
    cv2.createTrackbar("Threshold 2", "TrackBars", 0, 500, empty)           # Format: Title, window, start, stop, function
    while 1:
        imgContour = deepcopy(img)                                          # Duplicate Image
        t1 = cv2.getTrackbarPos("Threshold 1", "TrackBars")                 # Get trackbar value of particular trackbar (trackbar, window)
        t2 = cv2.getTrackbarPos("Threshold 2", "TrackBars")                 # Get trackbar value of particular trackbar (trackbar, window)
        imgCanny = cv2.Canny(imgBlur, t1, t2)                               # Requires img, threshold values
        getContours(imgCanny, imgContour)                                   # Draw all contours on imgContour
        imgStack = stackImages(1, ([img, imgCanny, imgContour]))            # Format: Scale, matrix of img example [img,img,img]
        cv2.imshow("ImageStack", imgStack)                                  # Display frame, arguments: Name of Window, img file
        cv2.waitKey(1)                                                      # Infinite delay else image closes instantly or show for specified ms

def facedetection(path):
    img = cv2.imread(path)                                                  # Read image
    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                         # Convert to particular color space - grayscale (bgr format cv2)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)                   # Values can be changed based on your image
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)          # 255 0 0 is blue, thickness - arguments

    cv2.imshow("Result", img)                                               # Display image, arguments: Name of Window, img file
    print(img.shape)                                                        # Displays size in pixels in height, width, no of channels (y, x, bgr) format
    cv2.waitKey(0)                                                          # Infinite delay else image closes instantly or show for specified ms

def videofacedetection():
    vid = cv2.VideoCapture(0)                                               # 0 - default webcam, 1 - next webcam
    vid.set(3, 640)                                                         # width as 640 px
    vid.set(4, 480)                                                         # height as 480 px
    vid.set(10, 100)                                                        # brightness as 100
    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    while 1:                                                                # Reading each frame one by one
        success, frame = vid.read()                                         # Boolean for success or failure and image frame file
        key = cv2.waitKey(1)
        if key == ord('q'):                                            # Close after q is pressed
            break
        if key == ord('s'):
            cv2.imwrite("Screenshots/Selfie.jpg", frame)
            # cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            # cv2.putText(frame, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,2, (0, 0, 255), 2)
            # cv2.imshow("Result", frame)
            # cv2.waitKey(500)
            print("Screenshot saved")

        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                   # Convert to particular color space - grayscale (bgr format cv2)
        faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)               # Values can be changed based on your image
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # 255 0 0 is blue, thickness - arguments
        cv2.imshow("Result", frame)                                          # Display image, arguments: Name of Window, img file


#imgcapture()
#imgCreate()
#videocapture()
#webcamcapture()
#imgGrayScale()
#imgBlur()
#imgCanny()
#cardread()
#imgMulti()
#colordetect()
#shapedetect()
#shapedetectcustom1Tracker("Images/shapes.png")
#shapedetectcustom2Tracker("Images/shapes.png")
#facedetection("Images/chad.png")
videofacedetection()
cv2.destroyAllWindows()