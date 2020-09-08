"""
@author  : MG
@Time    : 2020/9/8 13:34
@File    : plot_data.py
@contact : mmmaaaggg@163.com
@desc    : 用于
"""
from datetime import datetime
from vnpy.trader.constant import Exchange,Interval
from vnpy.trader.database import database_manager
import matplotlib.pyplot as plt

# Load history data
bars =database_manager.load_bar_data(
    symbol="a2005",
    exchange=Exchange.DCE,
    interval=Interval.MINUTE,
    start=datetime(2017, 4, 1),
    end=datetime(2019, 10, 30)
    )

# Generate x, y
y = []
for bar in bars:
    close_price = bar.close_price
    y.append(close_price)
x = list(range(1,len(y)+1))

# Show foto
plt.figure(figsize=(40, 20))
plt.plot(x, y)

plt.show()
