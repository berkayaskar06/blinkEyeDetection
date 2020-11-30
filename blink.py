#Kütüphaneler
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])
    C=dist.euclidean(eye[0],eye[3])

    #EAR Hesabı
    ear = (A+B)/(2.0*C)

    return ear



#Threshold
EYE_AR_THRESH =0.25
EYE_AR_CONSEC_FRAMES=3

#Counters
COUNTER = 0
TOTAL = 0

#init dlib
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\shape_predictor_68_face_landmarks.dat")#In Git


#Eye Recognize
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#start video stream
print ("[INFO] starting video stream thread...")
vid = cv2.VideoCapture(0)



while True:
    ret,frame = vid.read()
    if ret is False:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)
    for rect in rects:
        shape=predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        #left and right eye coordinates
        leftEye= shape[lStart:lEnd]
        rightEye= shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR+rightEAR)/2.0
        #convex hull for eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
        if ear<EYE_AR_THRESH:
            COUNTER+= 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                COUNTER = 0 
            cv2.putText(frame,"Blinks: {}".format(TOTAL),(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
            cv2.putText(frame,"EAR : {:.2f}".format(ear),(300,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.imshow("Frame",frame)
        key =cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            break


cv2.destroyAllWindows()





