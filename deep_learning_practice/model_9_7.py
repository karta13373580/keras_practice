import plotly
from plotly.graph_objects import Scatter,Layout
from plotly.offline import plot
import csv
import time
import pandas as pd
import twstock
import os

filepath = "twstockyear2018.csv"
if os.path.isfile(filepath):
    title = ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"]
    outputfile = open(filepath, "a", newline="", encoding="big5")
    outputwriter = csv.writer(outputfile)
    for i in range(12,13):
        stock = twstock.Stock("2317")
        stocklist = stock.fetch(2018,i)
        data = []
        for stock in stocklist:
            strdate = stock.date.strftime("%Y-%m-%d")
            li = [strdate, stock.capacity, stock.turnover,
                  stock.open, stock.high, stock.low, stock.close, stock.change, stock.transaction]
            data.append(li)
        """if i==1:
            outputwriter.writerow(title) """
        for dataline in (data):
                outputwriter.writerow(dataline)
        time.sleep(1)
    outputfile.close()
pdstock = pd.read_csv(filepath,encoding="big5")
pdstock["日期"] = pd.to_datetime(pdstock["日期"])
data1 = [
    Scatter(x = pdstock["日期"],y = ["收盤價"],name="收盤價"),
    Scatter(x = pdstock["日期"],y = ["最低價"],name="最低價"),
    Scatter(x = pdstock["日期"],y = ["最高價"],name="最高價")
]
plot({"data":data1,"layout":Layout(title="2018年個股統計圖")},auto_open=True)