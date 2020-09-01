import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from plotly.graph_objects import Scatter,Layout
from plotly.offline import plot
#資料處理
def load_data(df,sequence_length=10,split=0.8):
    data_all = np.array(df).astype("float64")
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data=[]
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i:i + sequence_length + 1])
    reshaped_data = np.array(data).astype("float64")
    x = reshaped_data[:,:-1]
    y = reshaped_data[:,-1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[:split_boundary] #2維
    train_y = y[:split_boundary] #1維

    test_x = x[split_boundary:]  #2維
    test_y = y[split_boundary:]  #1維

    return train_x, train_y, test_x, test_y, scaler
#模型建構
def build_model():
    model=Sequential()
    model.add(LSTM(
        input_shape=(10,1),units=256,unroll=False
    ))
    model.add(Dense(units=1))
    model.compile(loss="mse",optimizer="adam",metrics=["accuracy"])
    return model
def train_model(train_x, train_y, test_x, test_y):
    model = build_model()
    try:
        model.fit(train_x, train_y, batch_size=100, epochs=300, validation_split=0.1)
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    return predict,test_y

#主程式
pd.options.mode.chained_assignment = None
filename = "twstockyear2018.csv"
df = pd.read_csv(filename,encoding="big5")
ddprice = pd.DataFrame(df["收盤價"])
train_x, train_y, test_x, test_y, scaler =load_data(ddprice, sequence_length=10, split=0.8)
train_x = np.reshape(train_x,(train_x.shape[0],train_x.shape[1],1))
predict_y, test_y = train_model(train_x,train_y,test_x,test_y)

predict_y = scaler.inverse_transform([[i] for i in predict_y])
test_y = scaler.inverse_transform(test_y)

plt.plot(predict_y,"b:")
plt.plot(test_y,"r-")
plt.legend(["預測","收盤價"])
plt.show()

dd2 = pd.DataFrame({"predict":list(predict_y),"label":list(test_y)})
dd2["predict"] = np.array(dd2["predict"]).astype("float64")
dd2["label"] = np.array(dd2["label"]).astype("float64")

datax = [
    Scatter(y=dd2["predict"],name="預測"),
    Scatter(y=dd2["label"],name="收盤價")
]
plot({"data":datax,"layout":Layout(title="2018年個股預測圖")},auto_open=True)

