import cv2
import mediapipe as mp
import numpy as np
import time


class VirtualTrainer:

    def __init__(self, staticImg=False, mode=False, upBody=False, smooth=True, detectCon=0.9, trackCon=0.9):
        self.staticImg = staticImg
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.staticImg, self.mode, self.upBody, self.smooth, self.detectCon, self.trackCon)

        self.results = None

    def findPose(self, img, draw=True):
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)
            if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 1, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(radians * 180.0 / np.pi)
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 60, y2 + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        return angle

    def countReps(self, img, count, direc, wrong, direc2, l11, l12, l13, l21, l22, l23, a1, a2, w1, w2):
        angle1 = self.findAngle(img, l11, l12, l13)
        angle2 = self.findAngle(img, l21, l22, l23)
        per1 = np.interp(angle1, (a1, a2), (0, 100))
        per2 = np.interp(angle2, (a1, a2), (0, 100))
        if per1 == 100 and per2 == 100:
            if direc == 0:
                count += 0.5
                direc = 1
        if per1 == 0 and per2 == 0:
            if direc == 1:
                count += 0.5
                direc = 0
        wrong1 = np.interp(angle1, (w1, w2), (0, 100))
        wrong2 = np.interp(angle2, (w1, w2), (0, 100))
        if wrong1 == 100 and wrong2 == 100:
            if direc2 == 0:
                wrong += 1
                direc2 = 1
        if wrong1 == 0 and wrong2 == 0:
            if direc2 == 1:
                direc2 = 0
        cv2.putText(img, str(int(count)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(img, str(int(wrong - count)), (70, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        return count, direc, wrong, direc2

    def countRepsRev(self, img, count, direc, wrong, direc2, l11, l12, l13, l21, l22, l23, a1, a2, w1, w2):
        angle1 = self.findAngle(img, l11, l12, l13)
        angle2 = self.findAngle(img, l21, l22, l23)
        per1 = np.interp(angle1, (a1, a2), (0, 100))
        per2 = np.interp(angle2, (a1, a2), (0, 100))
        if per1 == 0 and per2 == 0:
            if direc == 0:
                count += 0.5
                direc = 1
        if per1 == 100 and per2 == 100:
            if direc == 1:
                count += 0.5
                direc = 0
        wrong1 = np.interp(angle1, (w1, w2), (0, 100))
        wrong2 = np.interp(angle2, (w1, w2), (0, 100))
        if wrong1 == 0 and wrong2 == 0:
            if direc2 == 0:
                wrong += 1
                direc2 = 1
        if wrong1 == 100 and wrong2 == 100:
            if direc2 == 1:
                direc2 = 0
        cv2.putText(img, str(int(count)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(img, str(int(wrong - count)), (70, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        return count, direc, wrong, direc2

def main():
    print("1. Bicep Curls")
    print("2. Shoulder Press")
    print("3. Squats")
    x = int(input("Your Choice- "))
    st = 0
    count, direc, wrong, direc2 = 0, 0, 0, 0
    l11, l12, l13, l21, l22, l23, a1, a2, w1, w2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if x == 1:
        st = "bc.mp4"
        l11, l12, l13, l21, l22, l23, a1, a2, w1, w2 = 11, 13, 15, 12, 14, 16, 40, 150, 90, 150
    elif x == 2:
        st = "sp.mp4"
        l11, l12, l13, l21, l22, l23, a1, a2, w1, w2 = 11, 13, 15, 12, 14, 16, 70, 160, 70, 100
    elif x == 3:
        st = "sq.mp4"
        l11, l12, l13, l21, l22, l23, a1, a2, w1, w2 = 23, 25, 27, 24, 26, 28, 80, 170, 120, 170
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0)
    pTime = 0
    trainer = VirtualTrainer()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (960, 540))
        img = trainer.findPose(img, False)
        lmList = trainer.findPosition(img, False)
        if len(lmList) != 0:
            if x == 1:
                count, direc, wrong, direc2 = trainer.countReps(img, count, direc, wrong, direc2, l11, l12, l13, l21, l22, l23, a1, a2, w1, w2)
            elif x == 2:
                count, direc, wrong, direc2 = trainer.countRepsRev(img, count, direc, wrong, direc2, l11, l12, l13, l21, l22, l23, a1, a2, w1, w2)
            elif x == 3:
                count, direc, wrong, direc2 = trainer.countReps(img, count, direc, wrong, direc2, l11, l12, l13, l21, l22, l23, a1, a2, w1, w2)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 266, 0), 3)
        cv2.imshow("AI Workout", img)
        cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
