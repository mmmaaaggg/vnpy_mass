"""
@author  : MG
@Time    : 2020/9/8 20:55
@File    : common.py
@contact : mmmaaaggg@163.com
@desc    : 用于数据加载数据合并过程中的公共函数
"""
import math
import os
import re
from datetime import datetime, timedelta, timezone
import pandas as pd
import config as _config  # NOQA
from vnpy.trader.constant import Exchange


PATTERN_INSTRUMENT_TYPE = re.compile(r'[A-Za-z]+(?=\d+$)')
# 交易所品种对照表参考链接
# http://www.khqihuo.com/spqh/1443.html
EXCHANGE_INSTRUMENTS_DIC = {
    Exchange.CFFEX: ("IF", "IC", "IH", "TS", "TF", "T"),
    Exchange.SHFE: ("CU", "AL", "ZN", "PB", "RU", "AU",
                    "FU", "RB", "WR", "AG", "BU", "HC",
                    "NI", "SN", "SP", "SS"),
    Exchange.CZCE: ("WH", "PM", "CF", "SR", "AT", "OI",
                    "RS", "RM", "RI", "FG", "ZC", "JR",
                    "MA", "LR", "SF", "SM", "CY", "AP",
                    "CJ", "UR"),
    Exchange.DCE: ("A", "B", "M", "Y", "C", "CS", "L",
                   "P", "V", "J", "JM", "I", "JD", "FB",
                   "BB", "PP", "RR", "EB", "EG"),
    Exchange.INE: ("SC", "NR"),
}

INSTRUMENT_EXCHANGE_DIC = {}
for _exchange, _inst_list in EXCHANGE_INSTRUMENTS_DIC.items():
    for _inst in _inst_list:
        INSTRUMENT_EXCHANGE_DIC[_inst] = _exchange


def get_file_iter(folder_path, filters=None):
    """
    文件列表迭代器，返回当前目录及子目录下的所有文件名及文件路径
    :param folder_path:
    :param filters: 文件后缀名筛选
    :return:
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isdir(file_path):
            yield from get_file_iter(file_path)
        else:
            if filters is None or os.path.splitext(file_name)[1] in filters:
                yield file_name, file_path


def generate_bar_dt(dt, minutes):
    date_only = datetime(dt.year, dt.month, dt.day).astimezone(timezone(timedelta(hours=8)))
    delta = dt.to_pydatetime() - date_only
    key = math.ceil(delta.seconds / 60 / minutes)
    by = date_only + timedelta(minutes=key * minutes)
    return by


def merge_df_2_minutes_bar(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    data_list = []
    for bar_dt, sub_df in df.groupby(df['datetime'].apply(lambda x: generate_bar_dt(x, minutes))):
        data_list.append({
            'datetime': bar_dt,
            'open_interest': sub_df['open_interest'].iloc[-1],
            'open_price': sub_df['open_price'].iloc[0],
            'high_price': sub_df['high_price'].max(),
            'low_price': sub_df['low_price'].min(),
            'close_price': sub_df['close_price'].iloc[-1],
            'volume': sub_df['volume'].sum(),  # 目前没有 volume
        })
    new_df = pd.DataFrame(data_list)
    return new_df


if __name__ == "__main__":
    pass
