from utils.HandTrackingModule import HandDetector
import mediapipe as mp
import numpy as np
import cv2
import time
import os

################################################################### Initializing Constants ###################################################################

IWidth, IHeight = 1280, 720
cTime = 0
pTime = 0

##############################################################################################################################################################

cap = cv2.VideoCapture(0)
Hand = HandDetector(colorP=(0, 0, 255), colorC=(0, 255, 0), circle_radius=2, thickness=cv2.FILLED)

cap.set(3, IWidth)
cap.set(4, IHeight)

################################################################### Read Images ###################################################################

imagePath = "E:/My_Model/3._Object_Detection/10. Finger Counter/Images"
myList = os.listdir(imagePath)
fingerImg = [cv2.imread(f"{imagePath}/{imgPath}") for imgPath in myList]

################################################################### Resize Image ###################################################################

resizedImg = []
for img in fingerImg:
    half = cv2.resize(img, (100, 100))
    resizedImg.append(half)
###################################################################################################################################################

while True:
    succ, frame = cap.read()

    frame[0:100, 0:100] = resizedImg[0]

    ################################################################### Fingers Tracking ###################################################################

    image = Hand.findHands(frame, draw=True)
    landMrks = Hand.findLandMarks(image, draw=False)

    fingerID = [4, 8, 12, 16, 20]

    if len(landMrks) != 0:
        fingers = []

        # Thumb
        cx, cy = landMrks[fingerID[0]][1], landMrks[fingerID[0] - 1][1]
        if cx > cy:
            fingers.append(1)
        else:
            fingers.append(0)

        # For Four Fingers
        for i in range(1, len(fingerID)):
            cx1, cy1 = landMrks[fingerID[i]][1], landMrks[fingerID[i]][2]
            cx2, cy2 = landMrks[fingerID[i] - 2][1], landMrks[fingerID[i] - 2][2]
            if cy1 < cy2:
                fingers.append(1)
            else:
                fingers.append(0)

            cv2.circle(image, (cx, landMrks[fingerID[0]][2]), 7, (255, 255, 0), cv2.FILLED)
            cv2.circle(image, (cx1, cy1), 7, (255, 255, 0), cv2.FILLED)
            cv2.circle(image, (cx2, cy2), 7, (255, 255, 0), cv2.FILLED)

        for i in range(5):
            num = fingers.count(1)
            frame[0:100, 0:100] = resizedImg[num]

    ################################################################### CalCulate FPS ###################################################################
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f"FPS: {int(fps)}", (50, 50), 1, 2, (255, 255, 0), 1)
    ################################################################### CalCulate FPS ###################################################################

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
