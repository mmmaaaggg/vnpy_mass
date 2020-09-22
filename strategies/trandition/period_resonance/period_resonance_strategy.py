"""
@author  : MG
@Time    : 2020/9/18 8:48
@File    : period_resonance_strategy.py
@contact : mmmaaaggg@163.com
@desc    : 用于多周期共振策略
首先研究 1分钟、5分钟周期共振，之后再扩展到多周期，甚至日级别共振
"""
from vnpy.app.cta_strategy import (
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
    CtaSignal,
    TargetPosTemplate
)
from config import logging

logger = logging.getLogger()


class MACDSignal(CtaSignal):
    """"""

    def __init__(self, fast_window: int, slow_window: int, signal_period: int, period: int = 30):
        """"""
        super().__init__()

        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_period = signal_period

        self.period = period
        self.bg = BarGenerator(self.on_bar, period, self.on_n_min_bar)
        self.am = ArrayManager(size=max(self.fast_window, self.slow_window, self.signal_period)+50)
        logger.info(f"fast_window, slow_window, signal_period, period="
                    f"{self.fast_window, self.slow_window, self.signal_period, self.period}")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.bg.update_bar(bar)

    def on_n_min_bar(self, bar: BarData):
        """"""
        self.am.update_bar(bar)
        if not self.am.inited:
            self.set_signal_pos(0)

        _, _, macd = self.am.macd(self.fast_window, self.slow_window, self.signal_period)

        if macd < -5:
            self.set_signal_pos(1)
        elif macd > 5:
            self.set_signal_pos(-1)
        else:
            # self.set_signal_pos(0)
            pass


class PeriodResonanceStrategy(TargetPosTemplate):
    """"""

    author = "MG"

    fast_window_1 = 6
    slow_window_1 = 26
    signal_period_1 = 9

    period_n = 30
    fast_window_n = 5
    slow_window_n = 14
    signal_period_n = 7

    signal_pos = {}

    parameters = ["fast_window_1", "slow_window_1", "signal_period_1",
                  "fast_window_n", "slow_window_n", "signal_period_n",
                  "period_n"]
    variables = ["signal_pos", "target_pos"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.write_log(f"setting={setting}")
        self.am = ArrayManager(size=max(self.fast_window_1, self.slow_window_1, self.signal_period_1)+50)
        self.macd_signal = MACDSignal(self.fast_window_n, self.slow_window_n, self.signal_period_n, self.period_n)

        self.signal_pos = {
            "macd_1": 0,
            "macd_n": 0,
        }

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        super().on_tick(tick)

        self.macd_signal.on_tick(tick)

        self.calculate_target_pos()

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        super().on_bar(bar)
        # 更新1分钟级别MACD信号
        self.am.update_bar(bar)
        if not self.am.inited:
            self.signal_pos['macd_1'] = 0

        dif, dea, macd = self.am.macd(
            self.fast_window_1, self.slow_window_1, self.signal_period_1)

        if macd < -5:
            self.signal_pos['macd_1'] = 1
            # self.write_log(f"{datetime_2_str(bar.datetime)} macd_1=1")
        elif macd > 5:
            self.signal_pos['macd_1'] = -1
            # self.write_log(f"{datetime_2_str(bar.datetime)} macd_1=-1")
        else:
            # self.set_signal_pos(0)
            pass
        self.macd_signal.on_bar(bar)

        self.calculate_target_pos()

    def calculate_target_pos(self):
        """"""
        self.signal_pos["macd_n"] = self.macd_signal.get_signal_pos()

        target_pos = 0
        for v in self.signal_pos.values():
            target_pos += v

        self.set_target_pos(target_pos)
        # if target_pos != 0:
        #     self.write_log(f"target_pos={target_pos}")

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        super().on_order(order)

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass

    def write_log(self, msg: str):
        super().write_log(msg)
        logger.info(msg)
