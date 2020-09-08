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
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import config as _config  # NOQA
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import database_manager
from vnpy.trader.object import TickData, BarData


def load_csv(file_path):
    """
    根据文件名，选择不同的格式进行解析，并插入tick数据库，同时合成分钟及小时数据
    :param file_path:
    :return:
    """
    _, file_name = os.path.split(file_path)
    # 大商所L1分笔行情
    if "".startswith('MFL1_TAQ_'):
        labels = [
            "Symbol", "ShortName", "SecurityID", "TradingDate", "TradingTime",
            "LastPrice", "HighPrice", "LowPrice", "TradeVolume", "TotalVolume",
            "LastVolume", "PreTotalPosition", "TotalPosition", "PrePositionChange", "",
            "TotalAmount", "TradeAmount", "PriceUpLimit", "PriceDownLimit", "PreSettlePrice",
            "PreClosePrice", "OpenPrice", "ClosePrice", "SettlePrice", "LifeLow",
            "Lifehigh", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
            "", "", "", "", "",
        ]
    ticks = []
    start = None
    count = 0
    with open(file_path, "r") as f:  # , encoding='utf-8'
        reader = csv.reader(f)
        for item in reader:
            localtime, instrument_id, trade_date, action_date, update_time, update_millisec, \
            last_price, volume, high_price, low_price, open_price, close_price, \
            avg_price, ask_price1, ask_vol1, bid_price1, bid_vol1, upper_limit_price, \
            lower_limit_price, open_interest, trun_over, pre_close, pre_open_interest, pre_settlement_price = item

            # generate datetime
            standard_time = f"{trade_date} {update_time}.{update_millisec}"
            dt = datetime.strptime(standard_time, "%Y%m%d %H:%M:%S.%f")

            # filter
            if time(15, 1) < dt.time() < time(20, 59):
                continue

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
                volume=float(volume),
                open_interest=float(open_interest),
                last_price=float(last_price),
                limit_up=float(upper_limit_price),
                limit_down=float(lower_limit_price),
                open_price=float(open_price),
                high_price=float(high_price),
                low_price=float(low_price),
                pre_close=float(pre_close),
                bid_price_1=float(bid_price1),
                bid_volume_1=float(bid_vol1),
                ask_price_1=float(ask_price1),
                ask_volume_1=float(ask_vol1),
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
                datetime=_['datetime'],
                interval=interval,
                volume=_["volume"],
                open_interest=_["open_interest"],
                open_price=_["open_price"],
                high_price=_["high_price"],
                low_price=_["low_price"],
                close_price=_["close_price"],
            ) for key, _ in interval_df.T.items()]
            database_manager.save_bar_data(bars)
            logger.info("插入 %s 数据%s - %s 总数量：%d", interval, start, end, len(bars))


if __name__ == "__main__":
    pass
