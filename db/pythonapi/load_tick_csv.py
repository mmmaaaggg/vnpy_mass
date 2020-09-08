"""
@author  : MG
@Time    : 2020/9/2 14:18
@File    : load_tick_csv_pythonpai.py
@contact : mmmaaaggg@163.com
@desc    : 用于加载 tick csv文件到 mysql 数据库供 vnpy 使用
【特别注意：】 vnpy 数据库没有主键约束，重复日期的数据可以反复导入，因此，避免程序重复执行 ！！！！

数据来源：
http://www.pythonpai.com/topic/4206/%E9%87%8F%E5%8C%96%E7%88%B1%E5%A5%BD%E8%80%85%E7%A6%8F%E5%88%A9%E8%B4%B4-%E9%87%8F%E5%8C%96%E4%BA%A4%E6%98%93%E4%BB%A3%E7%A0%81-%E5%B7%A5%E5%85%B7-2012-2020%E5%B9%B4%E6%9C%9F%E8%B4%A7%E5%85%A8%E5%93%81%E7%A7%8Dtick%E6%95%B0%E6%8D%AE%E5%85%B1%E4%BA%AB

代码来自 https://www.vnpy.com/forum/topic/1421-zai-ru-tickshu-ju-csvge-shi-dao-shu-ju-ku-zhong
在此基础上做了修改
数据没有表头，数据格式如下：
20200415085900,a2011,20200415,20200415,08:57:48,326,4149.00,0,-1.00,-1.00,-1.00,-1.00,0.00,4230.00,1,3870.00,7,4440.00,3860.00,355.00,0.00,4149.00,355.00,4150.00
数据文件字段
localtime (本机写入TICK的时间),
InstrumentID (合约名),
TradingDay (交易日),
ActionDay (业务日期),
UpdateTime （时间）,
UpdateMillisec（时间毫秒）,
LastPrice （最新价）,
Volume（成交量） ,
HighestPrice （最高价）,
LowestPrice（最低价） ,
OpenPrice（开盘价） ,
ClosePrice（收盘价）,
AveragePrice（均价）,
AskPrice1（申卖价一）,
AskVolume1（申卖量一）,
BidPrice1（申买价一）,
BidVolume1（申买量一）,
UpperLimitPrice（涨停板价），
LowerLimitPrice（跌停板价），
OpenInterest（持仓量）,
Turnover（成交金额）,
PreClosePrice (昨收盘),
PreOpenInterest (昨持仓),
PreSettlementPrice (上次结算价)
"""
import os
import csv
import logging
from datetime import datetime, time
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
    读取csv文件内容，并写入到数据库中
    当前文件没有考虑夜盘数据的情况，待有夜盘数据后需要对 trade_date 进行一定的调整
    """
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


def _test_csv_load():
    folder_path = r'd:\download\2020.4.15~2020.5.17.期货全市场行情数据\Data\20200415'
    file_path = os.path.join(folder_path, "../examples/au2004.csv")
    load_csv(file_path)


if __name__ == "__main__":
    # _test_csv_load()
    dir_path = r'd:\download\2020.4.15~2020.5.17.期货全市场行情数据\Data'
    run_load_csv(dir_path)
