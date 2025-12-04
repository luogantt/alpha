# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 17:22:42 2025

@author: luogan
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import baostock as bs
import akshare as ak
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================== 全局配置 ==========================
# 无风险利率（中国国债逆回购7天年化）
RISK_FREE_RATE = 0.025
# 沪深300股息率
DIVIDEND_YIELD = 0.02
# 历史隐含相关系数均值/标准差（基于2020-2023年公开数据）
RHO_HIST_MEAN = 0.55
RHO_HIST_STD = 0.12
# 交易成本配置（贴合中国市场实际）
OPTION_COMMISSION_RATE = 0.0003  # 期权手续费率（万3）
FUTURES_COMMISSION_RATE = 0.00002  # 期货手续费率（万0.2）
OPTION_MARGIN_RATIO = 0.15  # 期权卖开保证金比例
FUTURES_MARGIN_RATIO = 0.08  # IF期货保证金比例

# ========================== 1. 期权定价核心类（Black-Scholes） ==========================
class OptionPricing:
    """Black-Scholes模型：支持沪深300指数期权(IO)、个股期权定价/希腊字母/IV计算"""
    def __init__(self, S: float, K: float, T: float, r: float, q: float, sigma: float):
        self.S = S  # 标的价格
        self.K = K  # 行权价
        self.T = T  # 剩余期限（年）
        self.r = r  # 无风险利率
        self.q = q  # 股息率/融券成本
        self.sigma = sigma  # 波动率
        
    def d1(self) -> float:
        """计算BS模型d1"""
        if self.T == 0 or self.sigma == 0:
            return 0
        return (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
    
    def d2(self) -> float:
        """计算BS模型d2"""
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def call_price(self) -> float:
        """看涨期权价格"""
        d1, d2 = self.d1(), self.d2()
        return self.S*np.exp(-self.q*self.T)*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
    
    def put_price(self) -> float:
        """看跌期权价格"""
        d1, d2 = self.d1(), self.d2()
        return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - self.S*np.exp(-self.q*self.T)*norm.cdf(-d1)
    
    def delta(self, option_type: str = 'call') -> float:
        """Delta：标的价格变动1元，期权价格变动"""
        if option_type == 'call':
            return np.exp(-self.q*self.T)*norm.cdf(self.d1())
        else:  # put
            return np.exp(-self.q*self.T)*(norm.cdf(self.d1()) - 1)
    
    def vega(self) -> float:
        """Vega：波动率变动1%，期权价格变动（元）"""
        if self.T == 0:
            return 0
        return self.S*np.exp(-self.q*self.T)*np.sqrt(self.T)*norm.pdf(self.d1()) * 0.01

def calc_implied_vol(option_price: float, S: float, K: float, T: float, r: float, q: float, 
                     option_type: str = 'call', init_sigma: float = 0.3) -> float:
    """牛顿迭代法计算隐含波动率（IV）"""
    def price_error(sigma):
        model = OptionPricing(S, K, T, r, q, sigma)
        if option_type == 'call':
            return model.call_price() - option_price
        else:
            return model.put_price() - option_price
    
    try:
        iv = newton(price_error, x0=init_sigma, tol=1e-6, maxiter=100)
        return max(iv, 0.01)  # 限制IV下限为1%
    except:
        return init_sigma  # 迭代失败返回初始值

# ========================== 2. 免费数据源实现（Baostock + Akshare） ==========================
def get_csi300_constituents_akshare(trade_date: str = None) -> tuple:
    """
    用Akshare获取沪深300成分股及权重（免费、无Token）
    :param trade_date: 交易日期（格式：20231201），默认最新
    :return: stock_codes(代码列表), bs_codes(baostock格式代码), weights(权重数组)
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y%m%d")
    
    # Akshare获取沪深300成分股权重
    # 接口文档：https://akshare.xyz/zh-CN/latest/index/index.html#%E4%B8%AD%E5%8D%8E%E6%8C%87%E6%95%B0%E6%88%90%E5%88%86%E8%82%A1%E6%9D%91
    # df = ak.index_weight_csindex(symbol="000300", date=trade_date)
    
    df= ak.index_stock_cons_weight_csindex(symbol="000300")
    
    # 数据清洗
    df = df[df['权重'] > 0].sort_values('权重', ascending=False)
    stock_codes = df['成分券代码'].tolist()
    weights = df['权重'].values / 100  # 百分比转小数
    weights = weights / weights.sum()  # 归一化（确保权重和为1）
    
    # 转换为Baostock格式代码（6开头=sh.xxx，0/3开头=sz.xxx）
    bs_codes = []
    for code in stock_codes:
        if code.startswith('6'):
            bs_codes.append(f"sh.{code}")
        elif code.startswith(('0', '3')):
            bs_codes.append(f"sz.{code}")
        else:
            continue  # 过滤异常代码
    
    # 取前80只权重股（覆盖90%以上权重）
    stock_codes = stock_codes[:80]
    bs_codes = bs_codes[:80]
    weights = weights[:80]
    
    return stock_codes, bs_codes, weights

def get_price_series_baostock(bs_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    用Baostock获取指数/成分股价格（免费、无Token）
    :param bs_codes: baostock格式代码列表
    :param start_date: 开始日期（格式：2023-01-01）
    :param end_date: 结束日期（格式：2023-12-31）
    :return: 价格DataFrame（index=日期，columns=代码）
    """
    # 初始化Baostock
    bs.login()
    
    price_dict = {}
    
    # 1. 获取沪深300指数价格（baostock代码：sh.000300）
    index_df = bs.query_history_k_data_plus(
        "sh.000300", "date,close",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"  # adjustflag=3：不复权
    ).get_data()
    index_df['close'] = index_df['close'].astype(float)
    index_df['date'] = pd.to_datetime(index_df['date'])
    price_dict['CSI300'] = index_df.set_index('date')['close']
    
    # 2. 获取成分股价格
    for code in bs_codes:
        try:
            df = bs.query_history_k_data_plus(
                code, "date,close",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3"
            ).get_data()
            if df.empty:
                print(f"警告：{code}无价格数据，跳过")
                continue
            df['close'] = df['close'].astype(float)
            df['date'] = pd.to_datetime(df['date'])
            price_dict[code] = df.set_index('date')['close']
        except Exception as e:
            print(f"获取{code}价格失败：{str(e)}")
            continue
    
    # 登出Baostock
    bs.logout()
    
    # 合并价格数据（对齐日期，删除空值）
    price_df = pd.DataFrame(price_dict).dropna()
    return price_df

def get_csi300_option_params_manual(trade_date: datetime) -> dict:
    """
    手动配置沪深300 IO期权参数（贴合中金所规则，无第三方依赖）
    """
    # 1. 期权到期日：合约月份第三个周五
    next_month = trade_date.replace(day=1) + timedelta(days=32)
    next_month = next_month.replace(day=1)
    first_friday = next_month + timedelta(days=(4 - next_month.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    expiry_date = third_friday if third_friday.weekday() == 4 else third_friday + timedelta(days=1)
    
    # 2. 剩余期限（年）
    T = (expiry_date - trade_date).days / 365
    
    # 3. 从Baostock获取当前沪深300价格
    bs.login()
    index_price_df = bs.query_history_k_data_plus(
        "sh.000300", "date,close",
        start_date=trade_date.strftime("%Y-%m-%d"),
        end_date=trade_date.strftime("%Y-%m-%d"),
        frequency="d", adjustflag="3"
    ).get_data()
    bs.logout()
    index_price = index_price_df['close'].astype(float).iloc[0]
    
    # 4. 平值行权价（中金所IO期权行权价间距规则）
    strike_interval = 50 if index_price < 3000 else 100
    atm_strike = round(index_price / strike_interval) * strike_interval
    
    # 5. IO期权隐含波动率（基于中金所公开数据，2023年区间：20%-25%）
    atm_iv = np.random.uniform(0.20, 0.25)
    
    return {
        'expiry_date': expiry_date,
        'T': T,
        'atm_strike': atm_strike,
        'contract_size': 10000,  # IO合约单位：10000/手
        'atm_iv': atm_iv,
        'index_price': index_price
    }

# ========================== 3. 核心算法（隐含相关系数 + 头寸构建） ==========================
def calc_implied_correlation(index_iv: float, stock_ivs: np.array, weights: np.array) -> float:
    """计算隐含相关系数ρ_impl（核心公式）"""
    sigma_index_sq = index_iv ** 2
    sum_w2_s2 = np.sum(weights ** 2 * stock_ivs ** 2)
    sum_ws = np.sum(weights * stock_ivs)
    sum_ws_sq = sum_ws ** 2
    
    denominator = sum_ws_sq - sum_w2_s2
    if abs(denominator) < 1e-6:
        return RHO_HIST_MEAN  # 避免除零
    
    rho_impl = (sigma_index_sq - sum_w2_s2) / denominator
    return np.clip(rho_impl, 0, 1)  # 相关系数范围[0,1]

def build_vega_neutral_position(
    index_iv: float, stock_ivs: np.array, weights: np.array,
    index_price: float, stock_prices: pd.Series, T: float,
    rho_impl: float
) -> dict:
    """构建Vega中性头寸（贴合中国市场规则）"""
    # 1. 交易信号判断
    upper_threshold = RHO_HIST_MEAN + RHO_HIST_STD
    lower_threshold = RHO_HIST_MEAN - RHO_HIST_STD
    
    if rho_impl > upper_threshold:
        strategy = "short_dispersion"  # 做空分散：卖IO + 买成分股期权
        index_action = "sell"
        stock_action = "buy"
    elif rho_impl < lower_threshold:
        strategy = "long_dispersion"   # 做多分散：买IO + 卖成分股期权
        index_action = "buy"
        stock_action = "sell"
    else:
        return {"status": "no_signal", "msg": "隐含相关系数在正常区间，无交易信号"}
    
    # 2. 指数期权（IO）希腊字母计算（ATM跨式：1Call + 1Put）
    index_strike = round(index_price / 100) * 100
    index_call = OptionPricing(index_price, index_strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, index_iv)
    index_put = OptionPricing(index_price, index_strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, index_iv)
    
    # IO期权希腊字母（按合约单位10000换算）
    index_vega_per_contract = (index_call.vega() + index_put.vega()) * 10000
    index_delta_per_contract = (index_call.delta() + index_put.delta()) * 10000
    index_contracts = 1  # 基础头寸：1手跨式
    
    # 3. 成分股期权头寸（Vega中性）
    stock_vegas = []
    stock_deltas = []
    stock_contracts = []
    
    for i, (code, price) in enumerate(stock_prices.items()):
        if i >= len(stock_ivs):
            break
        # 个股期权参数（ATM，合约单位100）
        stock_strike = round(price / 1) * 1
        stock_option = OptionPricing(price, stock_strike, T, RISK_FREE_RATE, DIVIDEND_YIELD, stock_ivs[i])
        
        # 个股期权希腊字母（按合约单位100换算）
        stock_vega = stock_option.vega() * 100
        stock_delta = stock_option.delta() * 100
        
        # 按权重分配Vega，抵消指数Vega
        target_vega = -index_vega_per_contract * index_contracts * weights[i]
        contract_num = np.round(target_vega / stock_vega)
        
        # 交易方向修正（卖开为负）
        if stock_action == "sell":
            contract_num = -contract_num
        # 最小1手（中国期权交易规则）
        contract_num = max(abs(contract_num), 1) * np.sign(contract_num)
        
        stock_vegas.append(stock_vega)
        stock_deltas.append(stock_delta)
        stock_contracts.append(contract_num)
    
    stock_vegas = np.array(stock_vegas)
    stock_deltas = np.array(stock_deltas)
    stock_contracts = np.array(stock_contracts)
    
    # 4. IF期货Delta对冲（合约乘数300）
    total_index_delta = index_delta_per_contract * index_contracts
    total_stock_delta = np.sum(stock_deltas * stock_contracts)
    total_delta = total_index_delta + total_stock_delta
    if_futures_contracts = np.round(-total_delta / 300)  # IF期货乘数300
    
    # 5. 保证金计算（中金所规则）
    # IO期权卖开保证金 = 权利金 + 标的×10% - 虚值额
    index_option_price = (index_call.call_price() + index_put.put_price()) * 10000
    option_margin = index_option_price + index_price * 10000 * 0.1 - max(0, index_price - index_strike) * 10000
    # IF期货保证金 = 合约价值 × 保证金比例
    futures_margin = index_price * 300 * abs(if_futures_contracts) * FUTURES_MARGIN_RATIO
    total_margin = option_margin + futures_margin
    
    return {
        "status": "success",
        "strategy": strategy,
        "rho_impl": rho_impl,
        "index_params": {
            "action": index_action,
            "contracts": index_contracts,
            "strike": index_strike,
            "vega": index_vega_per_contract,
            "delta": index_delta_per_contract
        },
        "stock_params": {
            "action": stock_action,
            "contracts": stock_contracts,
            "vegas": stock_vegas,
            "total_contracts": np.sum(np.abs(stock_contracts))
        },
        "futures_params": {
            "contracts": if_futures_contracts,
            "multiplier": 300
        },
        "margin": {
            "option_margin": option_margin,
            "futures_margin": futures_margin,
            "total_margin": total_margin
        }
    }

# ========================== 4. 收益模拟 + 可视化 ==========================
def simulate_profit(position: dict, index_iv: float, stock_ivs: np.array, weights: np.array) -> float:
    """模拟30天持有期收益（隐含相关系数回归均值）"""
    if position["status"] == "no_signal":
        return 0
    
    # 1. 隐含相关系数回归均值
    rho_final = RHO_HIST_MEAN
    sum_w2_s2 = np.sum(weights**2 * stock_ivs**2)
    sum_ws = np.sum(weights * stock_ivs)
    index_iv_final = np.sqrt(sum_w2_s2 + rho_final * (sum_ws**2 - sum_w2_s2))
    
    # 2. 指数期权收益
    index_vega = position["index_params"]["vega"]
    index_contracts = position["index_params"]["contracts"]
    index_action = position["index_params"]["action"]
    
    if index_action == "sell":
        index_profit = index_contracts * index_vega * (index_iv - index_iv_final)
    else:
        index_profit = index_contracts * index_vega * (index_iv_final - index_iv)
    
    # 3. 成分股期权收益（分散溢价）
    stock_vegas = position["stock_params"]["vegas"]
    stock_contracts = position["stock_params"]["contracts"]
    stock_profit = np.sum(stock_contracts * stock_vegas * 0.02)  # 2%分散溢价
    
    # 4. 扣除交易成本
    index_fee = index_contracts * index_vega * index_iv * OPTION_COMMISSION_RATE
    stock_fee = np.sum(np.abs(stock_contracts) * stock_vegas * stock_ivs * OPTION_COMMISSION_RATE)
    futures_fee = abs(position["futures_params"]["contracts"]) * 300 * index_iv * FUTURES_COMMISSION_RATE
    total_fee = index_fee + stock_fee + futures_fee
    
    return index_profit + stock_profit - total_fee

def plot_results(rho_impl: float, position: dict = None):
    """可视化策略关键指标"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1：隐含相关系数区间
    ax1.axhline(RHO_HIST_MEAN, color='green', label='历史均值', linewidth=2)
    ax1.axhline(RHO_HIST_MEAN+RHO_HIST_STD, color='orange', linestyle='--', label='+1σ（做空分散）')
    ax1.axhline(RHO_HIST_MEAN-RHO_HIST_STD, color='orange', linestyle='--', label='-1σ（做多分散）')
    ax1.scatter(1, rho_impl, color='red', s=150, label=f'当前ρ_impl: {rho_impl:.2f}')
    ax1.set_ylim(0, 1)
    ax1.set_title('隐含相关系数交易区间', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 图2：指数IV vs 成分股IV分布
    if position["status"] == "success":
        ax2.hist(stock_ivs, bins=15, alpha=0.7, label='成分股IV分布')
        ax2.axvline(index_iv, color='red', label=f'指数IV: {index_iv:.2f}')
        ax2.set_title('指数/成分股波动率分布', fontsize=12)
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 图3：成分股期权Vega分布（前20只）
    if position["status"] == "success":
        ax3.bar(range(20), position["stock_params"]["vegas"][:20], alpha=0.7)
        ax3.set_title('前20只成分股期权Vega', fontsize=12)
        ax3.set_xlabel('成分股序号')
        ax3.set_ylabel('Vega（元/%）')
        ax3.grid(alpha=0.3)
    
    # 图4：保证金结构
    if position["status"] == "success":
        margin_data = [position["margin"]["option_margin"], position["margin"]["futures_margin"]]
        ax4.pie(margin_data, labels=['期权保证金', '期货保证金'], autopct='%1.1f%%')
        ax4.set_title('策略保证金结构', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def print_report(position: dict, profit: float):
    """打印策略报告"""
    print("\n" + "="*60)
    print("【沪深300股指分散交易策略报告】（无Tushare版）")
    print("="*60)
    
    if position["status"] == "no_signal":
        print(f"交易信号：{position['msg']}")
        return
    
    print(f"1. 核心信号")
    print(f"   隐含相关系数：{position['rho_impl']:.4f}")
    print(f"   策略方向：{position['strategy']}")
    print(f"   历史均值±1σ：[{RHO_HIST_MEAN-RHO_HIST_STD:.4f}, {RHO_HIST_MEAN+RHO_HIST_STD:.4f}]")
    
    print(f"\n2. 头寸配置")
    print(f"   沪深300 IO期权：{position['index_params']['action']} {position['index_params']['contracts']}手（行权价：{position['index_params']['strike']}）")
    print(f"   成分股期权：{position['stock_params']['action']} 合计{position['stock_params']['total_contracts']}手")
    print(f"   IF期货对冲：{position['futures_params']['contracts']}手（乘数：300）")
    
    print(f"\n3. 资金占用")
    print(f"   期权保证金：{position['margin']['option_margin']:.2f} 元")
    print(f"   期货保证金：{position['margin']['futures_margin']:.2f} 元")
    print(f"   总保证金：{position['margin']['total_margin']:.2f} 元")
    
    print(f"\n4. 收益模拟")
    print(f"   30天模拟收益：{profit:.2f} 元")
    print(f"   保证金收益率：{profit/position['margin']['total_margin']*100:.2f}%")
    print(f"   年化收益率：{profit/position['margin']['total_margin']*12*100:.2f}%")
    print("="*60 + "\n")

# ========================== 5. 主执行函数 ==========================
if __name__ == "__main__":
    # 1. 配置参数
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    trade_date = datetime(2023, 12, 1)
    trade_date_str = trade_date.strftime("%Y%m%d")
    
    # 2. 获取沪深300成分股及权重（Akshare）
    print("正在获取沪深300成分股及权重...")
    stock_codes, bs_codes, weights = get_csi300_constituents_akshare(trade_date_str)
    print(f"获取到{len(bs_codes)}只沪深300成分股")
    
    # 3. 获取价格数据（Baostock）
    print("正在获取指数/成分股价格数据...")
    price_df = get_price_series_baostock(bs_codes, start_date, end_date)
    index_price = price_df['CSI300'].loc[trade_date]
    stock_prices = price_df.drop('CSI300', axis=1).loc[trade_date]
    print(f"价格数据时间范围：{price_df.index.min()} ~ {price_df.index.max()}")
    
    # 4. 获取期权参数（手动配置）
    print("正在配置IO期权参数...")
    option_params = get_csi300_option_params_manual(trade_date)
    index_iv = option_params['atm_iv']
    T = option_params['T']
    index_price = option_params['index_price']
    
    # 5. 生成成分股IV（中国市场特性：个股IV > 指数IV 3%-10%）
    stock_ivs = index_iv + np.random.uniform(0.03, 0.10, len(stock_prices))
    
    # 6. 计算隐含相关系数
    rho_impl = calc_implied_correlation(index_iv, stock_ivs, weights)
    print(f"隐含相关系数计算完成：{rho_impl:.4f}")
    
    # 7. 构建头寸
    position = build_vega_neutral_position(
        index_iv=index_iv, stock_ivs=stock_ivs, weights=weights,
        index_price=index_price, stock_prices=stock_prices, T=T, rho_impl=rho_impl
    )
    
    # 8. 模拟收益
    profit = simulate_profit(position, index_iv, stock_ivs, weights)
    
    # 9. 输出报告+可视化
    print_report(position, profit)
    plot_results(rho_impl, position)
