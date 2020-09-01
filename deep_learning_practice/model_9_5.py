import csv
import pandas as pd
import twstock
import os
import matplotlib.pyplot as plt

filepath = "twtockmonth01.csv"

if not os.path.isfile(filepath):
  stock = twstock.Stock("2317")
  stocklist = stock.fetch(2018,1)
  title = ["日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數"]
  data=[]
  for stock in stocklist:
      strdate = stock.date.strftime("%Y-%m-%d")
      li = [strdate,stock.capacity,stock.turnover,
            stock.open,stock.high,stock.low,stock.close,stock.change,stock.transaction]
      data.append(li)

  outputfile = open(filepath,"w",newline="",encoding="big5")
  outputwriter = csv.writer(outputfile)
  outputwriter.writerow(title)
  for dataline in (data):
      outputwriter.writerow(dataline)
  outputfile.close()

pdstock = pd.read_csv(filepath,encoding="big5")
pdstock["日期"] = pd.to_datetime(pdstock["日期"])
pdstock.plot(kind="line",figsize=(12,6),x="日期",y=["收盤價","最低價","最高價"])
plt.show()