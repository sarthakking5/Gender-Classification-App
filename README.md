# Gender-Classification-App
The Gender Classification App is a web application developed on Flask and is integrated with a Pipeline Machine Learning Model.The images are processed using OpenCv and the necessary features are extracted and optimized using Principal Component Analysis.The Eigen images are then further used to train the Machine Learning model an SVM algorithm is used for the same.Lastly,the model is tuned with Grid Search Method for the best hyperparameters.

![002](https://user-images.githubusercontent.com/32523883/96698958-574e0b80-13ab-11eb-9fa3-880cb74ca225.PNG)

## Table of Contents

### 1)Data Processing

### 2)Machine Learning Model

### 3)Flask App


## Data Processing

This section deals with the processing of the images.Around 7000 images of each class (Male,Female) are converted into Grayscale and then cropped using a haar classifier.After this the images are then normalized,resized and ultimately are flattened.

![male_01](https://user-images.githubusercontent.com/32523883/96700254-df80e080-13ac-11eb-8eb2-37f686d57fb7.png)

## Machine Learning Model

The images are processed using OpenCv and the necessary features are extracted and optimized using Principal Component Analysis.The Eigen images are then further used to train the Machine Learning model an SVM algorithm is used for the same.Lastly,the model is tuned with Grid Search Method for the best hyperparameters.The model further predicts the uploaded image and classifies it as either male or female. 

![test](https://user-images.githubusercontent.com/32523883/96702939-e5c48c00-13af-11eb-899e-7c72953c50e7.jpg)

## Flask App

This section illustrates the Web Application with which the Pipeline Model is integrated it allows the user to simply upload an image and gives back the prediction.

![gc1](https://user-images.githubusercontent.com/32523883/96705409-e0b50c00-13b2-11eb-8a57-69b4b86805c7.png)
