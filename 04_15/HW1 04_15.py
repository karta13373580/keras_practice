# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:25:17 2020

@author: User
"""

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt, numpy as np

from keras import losses
import torch
import torch.nn.functional as F
from keras.models import Sequential
from keras.optimizers import *
import numpy as np

from keras.layers.core import Dense
x_train=np.ones((len(range(0,16000,1)),1))
y_train=np.ones((len(range(0,16000,1)),1))


for i in range(0,16000,1):
    p=(i-2000)/1000
    x_train[i]=p
    y_train[i]=1+np.sin(np.pi/4*p)
        
plt.figure(figsize=(8,8))
plt.plot(x_train,y_train,'g')
plt.show()

x_test=np.ones((len(range(0,16000,1)),1))
y_test=np.ones((len(range(0,16000,1)),1))


for i in range(0,16000,1):
    p=(i-2000)/1000
    x_test[i]=p-1
    y_test[i]=(1+np.sin(np.pi/4*p))/2
        
plt.figure(figsize=(8,8))
plt.plot(x_test,y_test,'g')
plt.show()


model1 = Sequential()
model1.add(Dense(32,input_dim=1,activation='sigmoid'))
model1.add(Dense(16,activation='linear'))
model1.add(Dense(16,activation='linear'))
model1.add(Dense(16,activation='linear'))
model1.add(Dense(1,activation='linear'))
model1.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=1e-1,decay=1e-5))
model1.fit(x_train,y_train,batch_size=80,epochs=32)
loss=model1.evaluate(x_test,y_test)
print("\nloss: %.10f" %(loss))
y_pred=model1.predict(x_test)

plt.figure(figsize=(8,8))
plt.plot(x_train,y_train,'g',x_test,y_pred,'r-.')
plt.show()
