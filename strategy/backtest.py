"""
@author  : MG
@Time    : 2020/9/13 19:03
@File    : backtest.py
@contact : mmmaaaggg@163.com
@desc    : 用于
"""
from ibats_utils.mess import str_2_date
from vnpy.app.cta_strategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Interval
from strategy.trandition.atr_rsi_strategy import AtrRsiStrategy

engine = BacktestingEngine()

engine.set_parameters(
    vt_symbol='rb2101',
    interval=Interval.MINUTE,
    start=str_2_date('2020-01-01'),
    rate=3e-5,  # 手续费
    slippage=0.001,  # 滑点
    size=1,  # 乘数
    pricetick=1,  # 最小价格变动
    capital=1000000,
)
engine.add_strategy(AtrRsiStrategy, setting={})
engine.run_backtesting()


if __name__ == "__main__":
    pass
