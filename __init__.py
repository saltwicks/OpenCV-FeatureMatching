import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
MIN_MATCH_COUNT = 10 #5 for images, 15 for big tin
MAX_MATCH_THRESH = 35 #35 for images, 45 for big tin
cap = cv2.VideoCapture(0)

searchImg = cv2.imread('Altoid/Top.jpg', 1)
searchImg = cv2.resize(searchImg, (0,0), fx=0.2, fy=0.2) 
img = cv2.imread('image.jpg', 0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(searchImg, None)
points = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    center = ((frame.shape[0] + searchImg.shape[0]), frame.shape[1]/3)
    itemCenter = (0,0)
    
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    good = []
    for m in matches:
        if m.distance < MAX_MATCH_THRESH:
            good.append(m)
    print len(good)
    if(len(good) > MIN_MATCH_COUNT):

        for m in good:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        
            h,w,_ = searchImg.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = pts
            if(M is not None):
                dst = cv2.perspectiveTransform(pts,M)
            frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            #itemCenter = getCenter(pts)
    
    img3 = cv2.drawMatches(searchImg, kp1, frame, kp2, good, None, flags=2)
    # Our operations on the frame come here
    #img3 = cv2.circle(img3, center , 100, 255, thickness=3, lineType=8, shift=0)

    cv2.imshow("Object Recognition", img3)
    #cv2.imshow("img", searchImg)
       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(.1)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


def getCenter(pts):
    return pts.mean(0)
    