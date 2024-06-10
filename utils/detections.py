import cvzone
import math
from utils.HandTrackingModule import HandDetector
import cv2
import numpy as np
import mediapipe as mp
import time
import os

class Detector:
    def __init__(self, videoPath, target_video_width=640,
                 mode=False, NumHands=2, ModelCompx=1, detectConf=0.5,
                 trackConf=0.5, colorP=(0, 255, 0), colorC=(255, 255, 255),
                 circle_radius=1, thickness=2, ):
        self.mode = mode
        self.NumHands = NumHands
        self.ModelCompx = ModelCompx
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.colorP = colorP
        self.colorC = colorC
        self.circle_radius = circle_radius
        self.thickness = thickness

        # tracking points
        self.fingerID = [4, 8, 12, 16, 20]

        self.videoPath = videoPath

        self.hand = HandDetector(self.mode, self.NumHands, self.ModelCompx, self.detectConf, self.trackConf,
                                 self.colorP, self.colorC, self.circle_radius, self.thickness)

        self.cam = cv2.VideoCapture(self.videoPath)

        self.cTime = 0
        self.pTime = 0

        # Get original video width and height
        self.original_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the aspect ratio
        aspect_ratio = self.original_width / self.original_height

        # Set the target width and height
        self.target_width = target_video_width
        self.target_height = int(self.target_width / aspect_ratio)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (self.target_width, self.target_height))

    # video will automatically turn off after closing website
    def __del__(self):
        return self.cam.release()

    def fingerImg(self):
        imagePath = "Images"
        resizedImg = []

        myList = os.listdir(imagePath)
        fingerImg = [cv2.imread(f"{imagePath}/{imgPath}") for imgPath in myList]

        for img in fingerImg:
            half = cv2.resize(img, (100, 100))
            resizedImg.append(half)
        return resizedImg

    def outputFrames(self):
        """
        This function is use to display the predictions on live video
        This function can be use when predicting on live fotage
        """
        # while True:
        succ, image = self.cam.read()
        resized_frame = cv2.resize(image, (self.target_width, self.target_height))
        image = self.hand.findHands(resized_frame, draw=True)

        print("resized Frame Size",resized_frame.shape)

        image[0:100, 0:100] = resized_frame[0][0]

        landMrks = self.hand.findLandMarks(image, draw=False)

        if len(landMrks) != 0:
            fingers = []

            # Thumb
            cx, cy = landMrks[self.fingerID[0]][1], landMrks[self.fingerID[0] - 1][1]
            if cx > cy:
                fingers.append(1)
            else:
                fingers.append(0)

            # For Four Fingers
            for i in range(1, len(self.fingerID)):
                cx1, cy1 = landMrks[self.fingerID[i]][1], landMrks[self.fingerID[i]][2]
                cx2, cy2 = landMrks[self.fingerID[i] - 2][1], landMrks[self.fingerID[i] - 2][2]
                if cy1 < cy2:
                    fingers.append(1)
                else:
                    fingers.append(0)

                cv2.circle(image, (cx, landMrks[self.fingerID[0]][2]), 7, (255, 255, 0), cv2.FILLED)
                cv2.circle(image, (cx1, cy1), 7, (255, 255, 0), cv2.FILLED)
                cv2.circle(image, (cx2, cy2), 7, (255, 255, 0), cv2.FILLED)

            resizedImg = self.fingerImg()
            for i in range(5):
                num = fingers.count(1)
                image[0:100, 0:100] = resizedImg[num]

        ################################################################### CalCulate FPS ###################################################################
        self.cTime = time.time()
        FPS = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(image, f"FPS: {int(FPS)}", (500,50), 1, 2, (255, 0, 255), 1, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        ################################################################# CalCulate FPS ################################################################################


        ####################################################################################################################
        self.out.write(image)
        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        return frame
