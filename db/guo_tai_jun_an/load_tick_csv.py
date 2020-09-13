"""
@author  : MG
@Time    : 2020/9/8 16:36
@File    : load_tick_csv.py
@contact : mmmaaaggg@163.com
@desc    : 用于加载国泰君安提供的tick数据
根据 新高频数据说明书20180613.pdf 提供的格式。不同交易所，不同行情基本的数据格式略有不同，需要分别处理
"""
import os
import csv
import logging
from datetime import datetime, time, timezone, timedelta
import pandas as pd
from queue import Queue, Empty
from threading import Thread
import config as _config  # NOQA
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import database_manager
from vnpy.trader.object import TickData, BarData

from db.common import get_file_iter, merge_df_2_minutes_bar, get_exchange

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.warning(r"""加载tick数据可能会非常占用数据库资源
请适度增加 tmp_table_size innodb_buffer_pool_size 大小
防止出现 (1206, 'The total number of locks exceeds the lock table size') 错误，推荐调整 MYSQL 参数：
tmp_table_size=2048M
innodb_buffer_pool_size=512M
另外，为了降低磁盘IO负担， sync_binlog 默认为 1 -> 调整为 0
Windows系统配置文件：c:\ProgramData\MySQL\MySQL Server 8.0\my.ini
""")


class JobWorker(Thread):
    def __init__(self, maxsize=20):
        super().__init__()
        self.job_queue = Queue(maxsize=maxsize)
        self.keep_waiting = True

    def run(self):
        while True:
            try:
                func, param = self.job_queue.get(timeout=60)
                func(param)
                self.job_queue.task_done()
                logger.info('%s 执行 %d 数据任务完成', func.__name__, len(param))
            except Empty:
                if self.keep_waiting:
                    logger.info('等待队列任务...')
                else:
                    break


def run_load_csv(folder_path=os.path.curdir, main_instrument_only=False, ignore_until_file_name=None):
    """
    遍历同一文件夹内所有csv文件，并且载入到数据库中
    """
    worker = JobWorker()
    worker.start()
    for n, (file_name, file_path) in enumerate(get_file_iter(
            folder_path, filters=[".csv"], ignore_until_file_name=ignore_until_file_name), start=1):
        logger.info("%d)载入文件：%s", n, file_path)
        load_csv(file_path, main_instrument_only, worker.job_queue)

    worker.keep_waiting = False
    worker.join()
    logger.info("所有任务完成")


def load_csv(file_path, main_instrument_only=False, job_queue: Queue = None):
    """
    根据文件名，选择不同的格式进行解析，并插入tick数据库，同时合成分钟及小时数据
    :param file_path:
    :param main_instrument_only: 仅接收主力合约
    :param job_queue: 是否使用任务队列
    :return:
    """
    _, file_name = os.path.split(file_path)
    # 大商所L1分笔行情
    if file_name.startswith('MFL1_TAQ_'):
        labels = [
            "Symbol", "ShortName", "SecurityID", "TradingDate", "TradingTime",
            "LastPrice", "HighPrice", "LowPrice", "TradeVolume", "TotalVolume",
            "LastVolume", "PreTotalPosition", "TotalPosition", "PrePositionChange", "TotalAmount",
            "TradeAmount", "PriceUpLimit", "PriceDownLimit", "PreSettlePrice", "PreClosePrice",
            "OpenPrice", "ClosePrice", "SettlePrice", "LifeLow", "LifeHigh",
            "AveragePrice01", "AveragePrice", "BidImplyQty", "AskImplyQty", "BuyOrSell",
            "SellPrice01", "BuyPrice01", "SellVolume01", "BuyVolume01", "SellPrice05",
            "SellPrice04", "SellPrice03", "SellPrice02", "BuyPrice02", "BuyPrice03",
            "BuyPriceO4", "BuyPrice05", "SellVolume05", "SellVolumeO4", "SellVolume03",
            "SellVolume02", "BuyVolume02", "BuyVolume03", "BuyVolume04", "BuyVolume05",
            "PreDelta", "Delta", "Change", "ChangeRatio", "Varieties",
            "ContinueSign", "Market", "UNIX", "OpenClose", "Amplitude",
            "VolRate", "OrderDiff", "OrderRate", "SellVOL", "BuyVOL",
            "PositionChange", "DeliverySettlePrice",
        ]
    else:
        raise ValueError(f"file_name='{file_name}' 目前不支持")

    ticks = []
    start = None
    count = 0
    symbol_idx = labels.index('Symbol')
    trading_time_idx = labels.index('TradingTime')
    volume_idx = labels.index('TradeVolume')
    open_interest_idx = labels.index('TotalPosition')
    last_price_idx = labels.index('LastPrice')
    upper_limit_price_idx = labels.index('PriceUpLimit')
    lower_limit_price_idx = labels.index('PriceDownLimit')
    open_price_idx = labels.index('OpenPrice')
    high_price_idx = labels.index('HighPrice')
    low_price_idx = labels.index('LowPrice')
    pre_close_idx = labels.index('PreClosePrice')
    bid_price1_idx = labels.index('BuyPrice01')
    bid_vol1_idx = labels.index('BuyVolume01')
    ask_price1_idx = labels.index('SellPrice01')
    ask_vol1_idx = labels.index('SellVolume01')
    exchange_idx = labels.index("Market")
    exchange = None
    with open(file_path, "r") as f:  # , encoding='utf-8'
        reader = csv.reader(_.replace('\x00', '') for _ in f)
        for item in reader:
            # generate datetime
            dt = datetime.strptime(item[trading_time_idx], "%Y-%m-%d %H:%M:%S.%f"
                                   ).astimezone(timezone(timedelta(hours=8)))

            # filter 剔除9点钟以前的数据， 以及 15:00 ~ 21：00 之间的数据
            if time(8, 0) <= dt.time() < time(9, 0) or time(15, 1) <= dt.time() < time(21, 0):
                continue

            instrument_id = item[symbol_idx]
            if exchange is None:
                instrument_type, exchange = get_exchange(instrument_id)
                if exchange is None:
                    try:
                        exchange = getattr(Exchange, item[exchange_idx])
                        logger.warning("当前品种 %s[%s] 不支持，需要更新交易所对照表后才可载入数据，使用数据中指定的交易所 %s",
                                       instrument_id, instrument_type, exchange)
                    except AttributeError:
                        logger.exception("当前品种 %s[%s] 不支持，需要更新交易所对照表后才可载入数据",
                                         instrument_id, instrument_type)
                        break

            # 仅接收主力合约
            if main_instrument_only and len(instrument_id) - len(instrument_type) == 4:
                break

            tick = TickData(
                symbol=instrument_id,
                datetime=dt,
                exchange=exchange,  # Exchange.SHFE
                volume=float(item[volume_idx]),
                open_interest=float(item[open_interest_idx]),
                last_price=float(item[last_price_idx]),
                limit_up=float(item[upper_limit_price_idx]),
                limit_down=float(item[lower_limit_price_idx]),
                open_price=float(item[open_price_idx]),
                high_price=float(item[high_price_idx]),
                low_price=float(item[low_price_idx]),
                pre_close=float(item[pre_close_idx]),
                bid_price_1=float(item[bid_price1_idx]),
                bid_volume_1=float(item[bid_vol1_idx]),
                ask_price_1=float(item[ask_price1_idx]),
                ask_volume_1=float(item[ask_vol1_idx]),
                gateway_name="DB",
            )
            ticks.append(tick)

            # do some statistics
            count += 1
            if not start:
                start = tick.datetime

        if count == 0:
            return

        end = tick.datetime
        if job_queue is None:
            database_manager.save_tick_data(ticks)
        else:
            job_queue.put((database_manager.save_tick_data, ticks))

        logger.info("插入 Tick 数据%s - %s 总数量：%d %s",
                    start, end, count, '' if job_queue is None else '加入任务队列')
        for n, (minutes, interval) in enumerate(zip([1, 60], [Interval.MINUTE, Interval.HOUR]), start=1):
            df = pd.DataFrame(
                [[
                    _.datetime, _.open_interest, _.last_price, _.volume
                ] for _ in ticks],
                columns=[
                    'datetime', 'open_interest', 'last_price', 'volume'
                ])
            interval_df = merge_df_2_minutes_bar(df, minutes)
            bars = [BarData(
                gateway_name="DB",
                symbol=instrument_id,
                exchange=exchange,
                datetime=_['datetime'].to_pydatetime(),  # Timestamp 数据可以通过 .tz_localize('Asia/Shanghai') 增加时区信息
                interval=interval,
                volume=_["volume"],
                open_interest=_["open_interest"],
                open_price=_["open_price"],
                high_price=_["high_price"],
                low_price=_["low_price"],
                close_price=_["close_price"],
            ) for key, _ in interval_df.iterrows()]
            if job_queue is None:
                database_manager.save_bar_data(bars)
            else:
                job_queue.put((database_manager.save_bar_data, bars))

            logger.info("插入 %s 数据%s - %s 总数量：%d %s",
                        interval, start, end, len(bars), '' if job_queue is None else '加入任务队列')


def _test_csv_load():
    folder_path = r'e:\TickData\MFL1_TAQ_202006'
    file_path = os.path.join(folder_path, "MFL1_TAQ_AG2007_202006.csv")
    load_csv(file_path)


if __name__ == "__main__":
    # _test_csv_load()
    dir_path = r'e:\TickData'
    run_load_csv(
        dir_path,
        # main_instrument_only=True,
        # ignore_until_file_name="MFL1_TAQ_NR2006_202006.csv"
    )
