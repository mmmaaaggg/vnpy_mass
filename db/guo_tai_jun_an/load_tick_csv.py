"""
@author  : MG
@Time    : 2020/9/8 16:36
@File    : load_tick_csv.py
@contact : mmmaaaggg@163.com
@desc    : 用于加载国泰君安提供的tick数据
根据 新高频数据说明书20180613.pdf 提供的格式。不同交易所，不同行情基本的数据格式略有不同，需要分别处理
"""
import math
import os
import csv
import re
import logging
from datetime import datetime, time, timezone, timedelta
import pandas as pd
import numpy as np
import config as _config  # NOQA
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import database_manager
from vnpy.trader.object import TickData, BarData

from db.common import get_file_iter, INSTRUMENT_EXCHANGE_DIC, PATTERN_INSTRUMENT_TYPE, merge_df_2_minutes_bar

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def run_load_csv(folder_path=os.path.curdir):
    """
    遍历同一文件夹内所有csv文件，并且载入到数据库中
    """
    for file_name, file_path in get_file_iter(folder_path, filters=[".csv"]):
        logger.info("载入文件：%s", file_path)
        load_csv(file_path)


def load_csv(file_path):
    """
    根据文件名，选择不同的格式进行解析，并插入tick数据库，同时合成分钟及小时数据
    :param file_path:
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
    with open(file_path, "r") as f:  # , encoding='utf-8'
        reader = csv.reader(f)
        for item in reader:
            # generate datetime
            dt = datetime.strptime(item[trading_time_idx], "%Y-%m-%d %H:%M:%S.%f"
                                   ).astimezone(timezone(timedelta(hours=8)))

            # filter
            if time(15, 1) <= dt.time() <= time(20, 59):
                continue

            instrument_id = item[symbol_idx]
            instrument_type = PATTERN_INSTRUMENT_TYPE.search(instrument_id).group()
            try:
                exchange = INSTRUMENT_EXCHANGE_DIC[instrument_type.upper()]
            except KeyError:
                logger.exception("当前品种 %s(%s) 不支持，需要更新交易所对照表后才可载入数据",
                                 instrument_type, instrument_id)
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
        database_manager.save_tick_data(ticks)
        logger.info("插入 Tick 数据%s - %s 总数量：%d", start, end, count)
        for n, (minutes, interval) in enumerate(zip([1, 60], [Interval.MINUTE, Interval.HOUR]), start=1):
            df = pd.DataFrame(
                [[
                    _.datetime, _.open_interest,
                    _.open_price, _.high_price, _.low_price, _.last_price, _.volume
                ] for _ in ticks],
                columns=[
                    'datetime', 'open_interest',
                    'open_price', 'high_price', 'low_price', 'close_price', 'volume'
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
            database_manager.save_bar_data(bars)
            logger.info("插入 %s 数据%s - %s 总数量：%d", interval, start, end, len(bars))


def _test_csv_load():
    folder_path = r'd:\download\MFL1_TAQ_202006'
    file_path = os.path.join(folder_path, "MFL1_TAQ_A2007_202006.csv")
    load_csv(file_path)


if __name__ == "__main__":
    # _test_csv_load()
    dir_path = r'd:\download\MFL1_TAQ_202006'
    run_load_csv(dir_path)
