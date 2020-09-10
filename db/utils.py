"""
@author  : MG
@Time    : 2020/9/10 8:45
@File    : utils.py
@contact : mmmaaaggg@163.com
@desc    : 用于
"""
from vnpy.trader.database import database_manager


def partition_bar_table():
    """
    bar 表分区语句生成并执行
    由于分区 key 必须要在主键之中，否则会报错：
    Error Code: 1503. A PRIMARY KEY must include all columns in the table's partitioning function
    :return:
    """
    from db.common import EXCHANGE_INSTRUMENTS_DIC
    partitions = []
    for exchange in EXCHANGE_INSTRUMENTS_DIC.keys():
        partitions.append(f"PARTITION {exchange.value} VALUES IN('{exchange.value}')")

    partition_str = ",\n    ".join(partitions)

    sql_str = """ALTER TABLE `dbbardata`
PARTITION BY LIST COLUMNS(exchange) 
SUBPARTITION BY KEY(symbol) SUBPARTITIONS 100
(
    """ + partition_str + "\n)"
    print(sql_str)
    # 手动执行sql 语句
    # meta = database_manager.class_bar._meta
    # meta.database.execute(sql_str)


if __name__ == "__main__":
    pass
