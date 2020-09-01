import numpy as np
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from  keras.layers.recurrent import SimpleRNN, LSTM


def show_image_labels_predictions(images,labels,predictions,start_id,num=10):
    plt.gcf().set_size_inches(12,14)
    if num>25:
        num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,i+1)
        ax.imshow(images[start_id],cmap="binary")
        if(len(predictions)>0):
            title = "ai = "+str(predictions[i])
            title += ("(0)"if predictions[i]==labels[i] else"(x)")
            title += "\nlabel = "+str(labels[i])
        else:
            title = "label = "+str(labels[i])
        ax.set_title(title,fontsize=12)
        ax.set_xticks([]);ax.set_yticks([])
        start_id += 1
    plt.show()

(train_feature, train_label),\
(test_feature, test_label) = mnist.load_data()

train_feature_vector = train_feature.reshape(len(train_feature),28,28).astype("float32")
test_feature_vector = test_feature.reshape(len(test_feature),28,28).astype("float32")

train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

train_label_onhot = np_utils.to_categorical(train_label)
test_label_onhot = np_utils.to_categorical(test_label)

model = Sequential()

""" model.add(SimpleRNN(
    input_shape=(28,28),
    units=256,
    kernel_initializer="normal",
    activation="relu",
    unroll=True
)) """
model.add(LSTM(
    input_shape=(28,28),
    units=256,
    unroll=True
))
model.add(Dropout(0.1))
model.add(Dense(
    units=10,
    kernel_initializer="normal",
    activation="softmax"
))

#訓練模型
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
train_history = model.fit(x=train_feature_normalize,
                          y=train_label_onhot,validation_split=0.2,
                          epochs=10,batch_size=200,verbose=2)

#評估準確率
scores = model.evaluate(test_feature_normalize,test_label_onhot)
print("\n準確率",scores[1])
prediction = model.predict_classes(test_feature_normalize)
show_image_labels_predictions(test_feature,test_label,prediction,0)

