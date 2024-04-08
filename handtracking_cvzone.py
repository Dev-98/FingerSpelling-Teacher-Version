import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
# from keras.models import load_model

cap = cv2.VideoCapture(0)
hand = HandDetector(maxHands=1)
classifiy = Classifier("my_model\converted_keras\keras_model.h5","my_model\converted_keras\labels.txt")

labels = ["A", "B", "C", "D", "E", "F","G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S","T", "U", "V", "W", "X", "Y"]
imgsize = 300
offset = 20

while True:
    success,video = cap.read()
    video = cv2.flip(video,1)

    hands,img = hand.findHands(video,False,False)
    # img = face(video)
    
    if hands:
        # If hands got detected,then crop the hand part from whole image 
        hand_Detected = hands[0]
        x,y,w,h = hand_Detected['bbox']

        # White canvas and cropping of the hand from image
        imgwhite = np.ones((imgsize,imgsize,3),np.uint8) * 255
        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        # imgCropShape = imgcrop.shape

        # Adjusting and Resizing cropped Image into the middle of canvas
        aspectRatio = h/w

        # When height is greater than width
        if aspectRatio > 1:
            k = imgsize/h
            wcal = math.ceil(k*w)

            imgResize = cv2.resize(imgcrop, (wcal,imgsize))
            wgap = math.ceil((imgsize-wcal)/2)

            # pasting hand onto canvas to avoid varying size of image
            imgwhite[:,wgap:wgap+wcal] = imgResize
            pred, index = classifiy.getPrediction(imgwhite)
            # print(pred,labels[index])
        
        # When width is greater than height
        else : 
            k = imgsize/w
            hcal = math.ceil(k*h)

            imgResize = cv2.resize(imgcrop, (imgsize, hcal))
            hgap = math.ceil((imgsize-hcal)/2)

            imgwhite[hgap:hcal+hgap, :] = imgResize
            pred, index = classifiy.getPrediction(imgwhite)
            # print(pred,labels[index])
            

        cv2.imshow("Canvas",imgwhite)

    cv2.imshow("Hand Tracking",img)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break


cap.release()
cv2.destroyAllWindows()
