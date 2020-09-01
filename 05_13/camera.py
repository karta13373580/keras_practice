# -*- coding: utf-8 -*-
"""
Created on Wed May 20 02:40:07 2020

@author: Lihen
"""
import cv2
import numpy as np
import os
from glob import glob
from PIL import Image

from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
import numpy as np
import  random
from sklearn.metrics import accuracy_score
import imageio
cv2.namedWindow("preview")
cam = cv2.VideoCapture(0)
cam.set(3,1920)
cam.set(4,720)


ret, image = cam.read()
k=0
model = load_model("model_2_4_3.h5")
while ret:
    cv2.imshow('preview',image)
    ret, image = cam.read()
    key=cv2.waitKey(33)
    if key == 27:
        break
    k+=1
    if(k==5):
        cv2.imwrite('SnapshotTest2.jpg',image)
cv2.destroyWindow('preview')
cam.release()


img_list=[]
img = cv2.imread('SnapshotTest2.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
img=np.reshape(img, (224,224,1))
img_list.append(img)
img_array=np.array(img_list)
#xtest = np.expand_dims(xtest, axis = 0)
#xtest = xtest/255
prediction=model.predict(img_array)
predict=np.argmax(prediction,axis=1)
print(predict)