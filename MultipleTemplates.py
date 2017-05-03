import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
MIN_MATCH_COUNT = 5
MAX_MATCH_THRESH = 20
cap = cv2.VideoCapture(0)

searchImg = cv2.imread('Altoid/Top.jpg', 1)
searchImg2 = cv2.imread('Altoid/Bottom.jpg', 1)
searchImg = cv2.resize(searchImg, (0,0), fx=0.2, fy=0.2) 
searchImg2 = cv2.resize(searchImg2, (0,0), fx=0.2, fy=0.2) 
img = cv2.imread('image.jpg', 0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(searchImg, None)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    good = []
    for m in matches:
        if m.distance < MAX_MATCH_THRESH:
            good.append(m)
    if(len(good) > MIN_MATCH_COUNT):
        print "The object is in the image"
          
    img3 = cv2.drawMatches(searchImg, kp1, frame, kp2, good, None, flags=2)
    # Our operations on the frame come here
       
    cv2.imshow("Object Recognition", img3)
    #cv2.imshow("img", searchImg)
       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(.1)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()