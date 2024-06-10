import cv2
import numpy as np
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, NumHands=2, ModelCompx=1, detectConf=0.5, trackConf=0.5,
                 colorP=(0, 255, 0), colorC=(255, 255, 255), circle_radius=1, thickness=2):
        self.mode = mode
        self.NumHands = NumHands
        self.ModelCompx = ModelCompx
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.colorP = colorP
        self.colorC = colorC
        self.circle_radius = circle_radius
        self.thickness = thickness

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.NumHands, self.ModelCompx, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpecP = self.mpDraw.DrawingSpec(color=self.colorP, circle_radius=self.circle_radius, thickness=self.thickness)
        self.drawSpecC = self.mpDraw.DrawingSpec(color=self.colorC)

    def findHands(self, img, draw=True):
        # 2. Convert our image BGR to RGB because mideapipe only needs RGB images
        ImgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 3. then pass the RGB image to process
        self.results = self.hands.process(ImgRGB)
        # print(results.multi_hand_landmarks)
        # Check the hand landmarks
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.drawSpecP, self.drawSpecC)
        return img

    def findLandMarks(self, img, HandNum=0, draw=False):
        """
        :param draw_con: this is to draw connections on landmarks
        :return: list of id and respective positions -> [id, cx,cy]
        """
        listLm = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[HandNum]
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                # calculate the center position of the landmark
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                listLm.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return listLm

    # def FingersUp(self, image, Hlandmarks):

        # finger1 = [4, 8, 12, 16, 20]
        # finger2 = [3, 7, 11, 15, 20]
        # finger3 = [2, 6, 10, 14, 19]
        # finger4 = [2, 5, 9, 13, 17]
        # finger5 = [2, 5, 9, 13, 17]

        # for i in range(len(fingerBottom)):
        #     cx1, cy1 = Hlandmarks[fingerTips[i]][1], Hlandmarks[fingerTips[i]][2]
        #     cx2, cy2 = Hlandmarks[fingerBottom[i]][1], Hlandmarks[fingerBottom[i]][2]
        #
        #     cv2.circle(image, (cx1, cy1), 7, (255, 0, 255), cv2.FILLED)
        #     cv2.circle(image, (cx2, cy2), 7, (255, 0, 255), cv2.FILLED)
        #     cv2.line(image, pt1=(cx1, cy1), pt2=(cx2, cy2), color=(255, 0, 0), thickness=1)



def main():
    cam = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0

    while True:
        success, img = cam.read()

        # create hand
        hand = HandDetector()
        img = hand.findHands(img, draw=True)
        listLmarks = hand.findLandMarks(img, HandNum=1, draw=True)

        if len(listLmarks) != 0:
            print(listLmarks[5])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, text=f"FPS: {str(int(fps))}", org=(50, 50), color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
