"""
@author  : MG
@Time    : 2020/9/14 10:16
@File    : mf_strategy.py
@contact : mmmaaaggg@163.com
@desc    : 基于单合约的多因子回归算法策略
优化目标位未来N日收益率
每M日更新一次
"""
import ffn  # NOQA
import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing, metrics
from ibats_common.backend.factor import get_factor
from sklearn.model_selection import train_test_split
from vnpy.app.cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    TargetPosTemplate,
)

BAR_ATTRIBUTES = [
    'open_price', 'high_price', 'low_price', 'close_price',
    'datetime', 'volume',
]


class MFStrategy(TargetPosTemplate):
    author = 'MG'

    trailing_percent = 0.8
    fixed_size = 1
    # 以未来 N 日范围计算 y 值作为训练目标
    target_n_bars = 5
    # 每 N 日重新训练一次
    retrain_pre_n_days = 5
    # 统计近 N 日的 bar 数据进行优化
    stat_n_days = 20
    # 弱学习器的最大迭代次数，
    # n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，
    # 一般选择一个适中的数值。默认是50。
    # 在实际调参的过程中，我们常常将n_estimators和learning_rate一起考虑。
    n_estimators = 50
    # 对于同样的训练集拟合效果，
    # 较小的ν意味着我们需要更多的弱学习器的迭代次数。
    # 通常用步长和迭代最大次数一起来决定算法的拟合效果。
    # 所以这两个参数n_estimators和learning_rate要一起调参。
    # 一般来说，可以从一个小一点的ν开始调参，默认是1。
    learning_rate = 0.01

    parameters = [
        "target_n_bars",
        "retrain_pre_n_days",
        "stat_n_days",
        "n_estimators",
        "learning_rate",
        "trailing_percent",
        "fixed_size",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """实例函数"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)
        self._hist_bar_list = []  # 已存入历史数据（临时存入），每次处理后会清空
        self._hist_bar_days = 0  # 已存入的 多少天的 bar 数据
        self._last_bar_date = None  # 上一根 bar 的日期
        self._is_new_day = False  # 判断是否当前bar为新的一天
        self.hist_bar_df = None  # 存储历史行情数据
        self._factor_df = None  # 存储历史行情的因子矩阵
        self.preparing = True  # 标示策略是否在准备当中（非交易状态）
        self.last_train_date = None
        self._current_bar = None  # 记录当前bar实例
        # 训练结果分类器以及归一化函数
        self.classifier, self.scaler = None, None

    def on_stack_bar(self, bar):
        """将 bar 数据存入临时列表，等待后续处理"""
        self._hist_bar_list.append(bar)
        curr_bar_date = bar.datetime.date()
        if self._last_bar_date is None:
            self._last_bar_date = curr_bar_date
        else:
            self._is_new_day = self._last_bar_date != curr_bar_date
            if self._is_new_day:
                self._hist_bar_days += 1

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.preparing = True
        self.write_log("策略初始化")
        self.load_bar(120 + 1000)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.preparing = False
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
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self._current_bar = bar
        super().on_bar(bar)
        self.cancel_all()

        # bar信息记入缓存
        self.on_stack_bar(bar)
        # if self.preparing:
        #     return

        # 生成因子数据
        self.generate_factors()
        # self.cancel_all()
        if self.last_train_date is None:
            # 从未进行过训练
            self.write_log(f"开始训练数据")
            self.retrain_data()

        if self.classifier is None:
            return

        target_position = self.predict()
        self.set_target_pos(target_pos=target_position)
        # # 平仓
        # if target_position <= 0 < self.pos:
        #     price = bar.close_price * (1 - self.trailing_percent / 100)
        #     self.sell(price=price, volume=abs(self.pos), stop=True)
        # elif self.pos >= 0 > target_position:
        #     price = bar.close_price * (1 + self.trailing_percent / 100)
        #     self.cover(price=price, volume=abs(self.pos), stop=True)
        #
        # # TODO: 开仓逻辑不够严谨，稍后修复
        # if self.pos < target_position:
        #     price = bar.close_price * (1 - self.trailing_percent / 100)
        #     self.buy(price=price, volume=self.fixed_size)
        # elif target_position < self.pos:
        #     price = bar.close_price * (1 - self.trailing_percent / 100)
        #     self.short(price=price, volume=self.fixed_size)

        # self.put_event()

    def retrain_data(self, force_train=False):
        """
        对收集的行情历史行情数进行因子整理，训练
        默认情况下只在某日行情结束，新bar生成时才开始判断是否训练，除非 force_train==True
        :param force_train:
        :return:
        """
        if not (self._is_new_day or force_train):
            # 只有每次换日时才重新训练
            return
        if self._factor_df.shape[0] >= 1000 and self._hist_bar_days < self.retrain_pre_n_days:
            # 每 N 天重新训练一次
            return

        # 生成 y 值
        # 测试例子
        # df = pd.DataFrame(np.linspace(1, 2) * 10 + np.random.random(50),
        #                   index=pd.date_range('2020-01-01', periods=50),
        #                   columns=['close_price'])
        factor_df = self._factor_df
        y_s = self.hist_bar_df['close_price'].rolling(
            window=5).apply(lambda x: x.calc_calmar_ratio())
        # 剔除无效数据
        is_available = ~(np.isinf(y_s) | np.isnan(y_s) | np.any(np.isnan(factor_df), axis=1))
        x_arr = self.hist_bar_df[is_available].to_numpy()
        y_arr = y_s[is_available]
        # 生成 -1 1 分类结果
        y_arr[y_arr > 0] = 1
        y_arr[y_arr <= 0] = -1
        x_train_arr, x_test_arr, y_train_arr, y_test_arr = train_test_split(
            x_arr, y_arr, test_size=0.3)

        # Train classifier
        scaler = preprocessing.MinMaxScaler()
        clf = ensemble.AdaBoostClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate)

        x_train_trans_arr = scaler.fit_transform(x_train_arr)
        # print 'type(X_train_trans),X_train_trans[:5,:]\n%s\n%s'%(type(X_train_trans),X_train_trans[:5,:])
        # print 'type(Y_train),Y_train.head()\n%s\n%s'%(type(Y_train),Y_train.head())
        clf.fit(x_train_trans_arr, y_train_arr)
        # 交叉检验
        y_pred = clf.predict(x_train_trans_arr)
        print('Accuracy on train set = {:.2f}%'.format(metrics.accuracy_score(y_train_arr, y_pred) * 100))
        x_test_trans = scaler.transform(x_test_arr)
        y_pred = clf.predict(x_test_trans)
        y_pred_prob = clf.predict_proba(x_test_trans)
        print('Accuracy on test set = {:.2f}%'.format(metrics.accuracy_score(y_test_arr, y_pred) * 100))
        print('Log-loss on test set = {:.5f}'.format(metrics.log_loss(y_test_arr, y_pred_prob)))
        self.classifier = clf
        self.scaler = scaler
        self.last_train_date = self._current_bar.datetime.date()

    def predict(self) -> int:
        """预测数据返回 -1空头，0，平仓，1多头"""
        factors = self._factor_df.iloc[-1:, :].to_numpy()
        x_trans = self.scaler.transform(factors)
        y_pred = self.classifier.predict(x_trans)
        return y_pred[0]

    def generate_factors(self):
        """整理缓存数据，生成相应的因子"""
        print("生成因子数据")
        self.write_log("生成因子数据")
        df = pd.DataFrame(
            [{key: getattr(_, key) for key in BAR_ATTRIBUTES}
             for _ in self._hist_bar_list]).set_index('datetime')
        df.index = pd.to_datetime(df.index)
        # 重置缓冲区状态
        self._hist_bar_list = []
        self._hist_bar_days = 0
        # 扩展 hist_bar_df
        if self.hist_bar_df is None:
            self.hist_bar_df = df
        else:
            self.hist_bar_df = self.hist_bar_df.append(df).sort_index()
        # 生成因子
        self._factor_df = get_factor(
            self.hist_bar_df,
            ohlcav_col_name_list=['open_price', 'high_price', 'low_price', 'close_price', None, 'volume'])

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        super().on_order(order)


if __name__ == "__main__":
    pass
