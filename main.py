import cv2
import numpy as np
import mediapipe as mp

#######################
brushThickness = 7
eraserThickness = 50
########################


# drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

i = 3
flag = 0

while True:

    success, img = cap.read()         #camera image
    img = cv2.flip(img, 1)

    # img = detector.findHands(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #hands detection
    res = hands.process(imgRGB)
    lmList = []
    if res.multi_hand_landmarks:
        for handLMS in res.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                lmList.append([id, cx, cy])
                if id:
                    cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)


    colorList = [(102, 0, 204), (0, 128, 255), (255, 51, 51), (153, 0, 153), (51, 255, 255), (18, 222, 150)]
    n = len(colorList)

    if len(lmList) != 0:

        # print(lmList)

        x1, y1 = lmList[8][1:]           #index finger
        x2, y2 = lmList[12][1:]          #middle finger


        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        for id in range(1, 5):                                          # fingers status
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)


        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:       #all fingers up
            xp, yp = 0, 0
            print("Erase Mode")

            drawColor = (0, 0, 0)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            flag = 0



        elif (fingers[0] == 1) and (fingers[1] == 0) and (fingers[2] == 0) and fingers[3] == 0:     #index finger up
            # drawColor=(255,0,255)

            cv2.circle(img, (x1, y1), 10, colorList[i], cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # if lmList[tipIds[2]][2] < lmList[tipIds[2] - 2][2]:
            #     xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), colorList[i], brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), colorList[i], brushThickness)

            xp, yp = x1, y1
            flag = 0

        elif fingers[0] and fingers[1]:
            xp, yp = 0, 0
            flag = 0

        elif fingers[0] == 1 and fingers[3] == 1 and fingers[1] == 0 and fingers[2] == 0:
            if (flag == 0):
                i = (i + 1)
                i = i % n
                flag = 1
            print("detected")
            # color=colorList[i]

        elif fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 1:
            imgCanvas = np.zeros((480, 640, 3), np.uint8)

        # else:
        #     continue

        # cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)


    # print(img.shape)
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    # print(_)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    print(imgInv.shape)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)