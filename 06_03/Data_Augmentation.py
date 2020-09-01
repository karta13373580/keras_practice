import os
from glob import glob
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
import cv2

picSize=224

def get_data(c):
    
    #時尚辨識
    data=glob(os.path.join("fashion_train\\"+str(c)+"\\","*.*"))
    
    #手寫辨識
    # data=glob(os.path.join("handwrite_train\\"+str(c)+"\\","*.*"))
    data=np.sort(data)
    st1=0
    nd1=data.size
    x0=np.zeros((len(range(st1,nd1,1)),picSize,picSize,3),dtype='uint8')
    li1=[i for i in range(data.size)]
    j=random.sample(li1,nd1)
    k=-1
    for i in range(st1,nd1,1):
        image=Image.open(data[j[i]])
        a=np.asarray(image)
#        a=cv2.resize(im,dsize=(picSize,picSize),interpolation=cv2.INTER_CUBIC)
        b=a.reshape((picSize,picSize,3))
        k=k+1
        x0[k]=b
    x_data=x0
    return x_data.astype('uint8')

aug=ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect'
        )

for c in range(0,10):
    total=0
    x_dat=get_data(c)
    x_dat = (np.expand_dims(x_dat, axis=0))
    
    #時尚辨識
    for image in aug.flow(x_dat,batch_size=1,save_to_dir=os.path.join("fashion_mnist\\"+str(c)+"\\"),save_prefix="image",save_format="jpg"):
    
    #手寫辨識
    # for image in aug.flow(x_dat,batch_size=1,save_to_dir=os.path.join("handwrite_mnist\\"+str(c)+"\\"),save_prefix="image",save_format="jpg"):
        total+=1
        if total>=len(x_dat):
            break
print("it is done")
