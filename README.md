# Gender-Classification-App
The Gender Classification App is a web application developed on Flask and is integrated with a Pipeline Machine Learning Model.The images are processed using OpenCv and the necessary features are extracted and optimized using Principal Component Analysis.The Eigen images are then further used to train the Machine Learning model an SVM algorithm is used for the same.Lastly,the model is tuned with Grid Search Method for the best hyperparameters.

## Table of Contents

### 1)Data Processing

### 2)Machine Learning Model

### 3)Flask App


## Data Processing

This section deals with the processing of the images.Around 7000 images of each class (Male,Female) are converted into Grayscale and then cropped using a haar classifier.After this the images are then normalized,resized and ultimately are flattened.
