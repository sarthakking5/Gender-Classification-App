import numpy as np
import cv2
from glob import glob


femalepath=glob("D:/data science/Module-2/data/female/*.jpg")
malepath=glob("D:/data science/Module-2/data/male/*.jpg")

haar=cv2.CascadeClassifier('D:/data science/Module-1/data/haarcascade_frontalface_default.xml')

def extract_images(path,gender,i):
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=haar.detectMultiScale(gray,1.5,5)
    for x,y,w,h in faces:
        roi=img[y:y+h,x:x+w]
        if gender=='male':
            cv2.imwrite('D:/data science/Module-2/cropped/male_crop/{}_{}.png'.format(gender,i),roi)
        else:
            cv2.imwrite('D:/data science/Module-2/cropped/female_crop/{}_{}.png'.format(gender, i), roi)




def process(malepath):
    for i,path in enumerate(malepath):
        try:
            extract_images(path,"male",i)
            print("INFO:{}/{} processed succesfully".format(i,len(malepath)))
        except:
            print("INFO:{}/{} cannnot be processed".format(i, len(malepath)))


process(malepath)


