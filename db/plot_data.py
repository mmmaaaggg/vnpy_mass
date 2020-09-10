"""
@author  : MG
@Time    : 2020/9/8 13:34
@File    : plot_data.py
@contact : mmmaaaggg@163.com
@desc    : 用于将数据 plot
"""
from datetime import datetime
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import database_manager
import matplotlib.pyplot as plt


def plot_bar(symbol="AG2007", exchange=Exchange.SHFE, interval=Interval.MINUTE,
             start=datetime(2019, 4, 1), end=datetime(2020, 10, 30),
             label_count=15, fig_size=(16, 6), label_rotation=15, time_format='%Y-%m-%d %H:%M:%S'):
    # Load history data
    bars = database_manager.load_bar_data(
        symbol=symbol, exchange=exchange,
        interval=interval, start=start, end=end)

    # Generate x, y
    x = [bar.datetime for bar in bars]
    y = [bar.close_price for bar in bars]

    # Show plot
    y_len = len(y)
    xticks = list(range(0, y_len, y_len // label_count))
    xlabels = [x[_].strftime(time_format) for _ in xticks]
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=label_rotation)
    plt.plot(y)
    plt.title(f"{symbol} {interval.value} {min(x).strftime(time_format)}~{max(x).strftime(time_format)}")
    plt.legend([symbol])
    plt.show()


if __name__ == '__main__':
    plot_bar()
