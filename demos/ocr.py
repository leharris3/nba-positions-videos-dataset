#omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam

import easyocr
import numpy as np 

from ultralytics import YOLO
from utilities.text_extraction.detect_roi import *


# Predict on an example.
def get_pts(img_path): 
    model = YOLO(MODEL_PATH)
    x = model(img_path, save=True)
    lstx = []
    lsty= []

    lst = x[0]

    for i in range(len(lst)):
        lst1 = lst[i].boxes.xyxy.numpy()[0]
        for j in range(len(lst1)):
            if (j%2 == 0):
                lstx.append(lst1[j])
            else:
                lsty.append(lst1[j])


    maxx = int(max(lstx))
    minx = int(min(lstx))
    maxy = int(max(lsty))
    miny = int(min(lsty))
    return maxx,minx,maxy,miny

def text(vid_path):
    reader = easyocr.Reader(['en'])
    cap = cv2.VideoCapture(vid_path)
    i = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while 1:
        ret, frame = cap.read()

        if frame is None:
            break
        height, width, layers = frame.shape
        if i == 0:
            cv2.imwrite("img_src.png", frame)
            x1,x2,y1,y2 = get_pts("img_src.png")
            print(x1,y1,x2,y2)
            pts = np.array([[[x1, y1],
                             [x2, y1],\
                             [x2, y2], \
                             [x1, y2]]])
            out = cv2.VideoWriter("masked.MOV", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(mask, pts=np.int32([pts]), color=(255, 255, 255))

        # apply the mask
        masked_image = cv2.bitwise_and(frame, mask)
        #gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        i += 1
        img = frame.copy()
        img = cv2.resize(img, (1280,736))
        cv2.imwrite("img.jpg", masked_image)

        if (i == 1 or i%1 == 0):
            result = reader.readtext("img.jpg")
            spacer = 100
            for detection in result:
                top_left = tuple(detection[0][0])
                bottom_right = tuple(detection[0][2])
                text = detection[1]
                img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
                img = cv2.putText(img, text, top_left, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                spacer += 15
                print(text)
                if ":" in text or ";" in text:
                    ptext = text
                    pt_left = top_left

        img = cv2.putText(img, ptext, pt_left, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("img", img)
        cv2.waitKey(1)
        out.write(img)
    out.release()



if __name__ == "__main__":
    text("test_25hz.mp4")
