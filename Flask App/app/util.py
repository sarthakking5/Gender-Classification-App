import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn

#loading models
haar=cv2.CascadeClassifier('D:/data science/Module-2/model/haarcascade_frontalface_default.xml')
mean=pickle.load(open('D:/data science/Module-2/model/mean_preprocess.pickle','rb'))
model_svm=pickle.load(open('D:/data science/Module-2/data/model_svm.pickle','rb'))
model_pca=pickle.load(open('D:/data science/Module-2/data/pca_50.pickle','rb'))

gender_pre=['Male','Female']
font=cv2.FONT_HERSHEY_SIMPLEX

def pipeline_model(path,filename,color='bgr'):
    #read image
    img=cv2.imread(path)
    #convert to grayscale
    if color=='bgr':
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #step 3 crop using haar classifier
    faces=haar.detectMultiScale(gray,1.5,3)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi=gray[y:y+h,x:x+w]
        #step4 Normalization(0-1)
        roi=roi/255.0
        #step5:Resize Image(100,100)
        if roi.shape[1]>100:
            roi_resize=cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize=cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
         #step6 Flatenning(1x10,000)
        roi_reshape=roi_resize.reshape(1,10000)
        #step7 subtract with mean
        roi_mean=roi_reshape-mean
        #step 8 get eigen image
        eigen_image=model_pca.transform(roi_mean)
        #step 9 pass to ml model
        results=model_svm.predict_proba(eigen_image)[0]
        #step10
        predict=results.argmax()
        score=results[predict]
        #step 11
        text='%s:%0.2f'%(gender_pre[predict],score)
        cv2.putText(img,text,(x,y),font,1,(0,255,0),2)
    cv2.imwrite('./static/predict/{}'.format(filename),img)
   