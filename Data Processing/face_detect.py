import numpy as np
import cv2


haar=cv2.CascadeClassifier('D:/data science/Module-1/data/haarcascade_frontalface_default.xml')

def face_detect(img):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=haar.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        #face_crop=img[x:x+w,y:y+h]
        #cv2.imwrite('D:/data science/Module-1/data/1.png',face_crop)
    return img



cap=cv2.VideoCapture('D:/data science/Module-1/data/video.mp4')

while True:
    ret,frame=cap.read()
    if ret==False:
        break

    frame=face_detect(frame)

    cv2.imshow('detect',frame)
    if cv2.waitKey(40)==27:
        break

cv2.destroyAllWindows()
cap.release()
