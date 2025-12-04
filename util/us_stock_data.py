





import requests
import pandas as pd
from dotenv import load_dotenv
import os

# 加载环境变量（.env文件中写入 POLYGON_API_KEY=你的密钥）
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io/v2/aggs"

def get_kline(ticker, interval="1/day", start_date="2023-01-01", end_date="2023-12-31"):
    """
    获取不同周期K线（修正后支持所有Polygon合法周期）
    :param ticker: 股票代码（如AAPL）
    :param interval: 周期（合法格式：1/minute、5/minute、1/hour、1/day、1/week、1/month、1/quarter、1/year）
    :param start_date: 开始日期（YYYY-MM-DD）
    :param end_date: 结束日期（YYYY-MM-DD）
    :return: DataFrame格式K线数据
    """
    # 验证周期格式（可选，提前规避错误）
    valid_units = ["minute", "hour", "day", "week", "month", "quarter", "year"]
    unit = interval.split("/")[-1]
    if unit not in valid_units:
        raise ValueError(f"无效周期单位！支持的单位：{valid_units}")
    
    url = f"{BASE_URL}/ticker/{ticker}/range/{interval}/{start_date}/{end_date}"
    params = {
        "adjusted": True,  # 复权数据（必开，修正分红/拆股影响）
        "limit": 5000,     # Free Tier单次最多返回5000条
        "apiKey": 'eZ81KlwHnTGygQVn60pIQ5MmihYzuPyu'
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("status") != "OK":
        raise ValueError(f"接口报错：{data.get('error')}")
    
    # 解析数据并格式化
    df = pd.DataFrame(data["results"])
    df.columns = ["时间戳", "开盘价", "最高价", "最低价", "收盘价", "成交量", "成交额", "换手率"]
    df["日期时间"] = pd.to_datetime(df["时间戳"], unit="ms")  # 时间戳转可读格式
    # 筛选常用字段，按时间排序
    df = df[["日期时间", "开盘价", "最高价", "最低价", "收盘价", "成交量", "成交额"]].sort_values("日期时间")
    return df

# ---------------------- 测试不同周期（修正后无报错）----------------------
# 1. 分时数据（1分钟线，2023-01-03当天）
df_minute = get_kline("AAPL", interval="1/minute", start_date="2025-11-03", end_date="2025-12-03")
print("AAPL 2023-01-03 1分钟线（前5条）：")
print(df_minute.head())

# 2. 5分钟线（2023-01-03当天）
df_5min = get_kline("AAPL", interval="1/minute", start_date="2025-11-03", end_date="2025-12-03")
print("\nAAPL 2023-01-03 5分钟线（前5条）：")
print(df_5min.head())

# 3. 小时线（2023-01-01至2023-01-07）
df_hour = get_kline("AAPL", interval="1/minute", start_date="2025-11-03", end_date="2025-12-03")
print("\nMSFT 2023-01-01至2023-01-07 1小时线（前5条）：")
print(df_hour.head())

# 4. 季度线（2020-01-01至2023-12-31）
df_quarter = get_kline("AAPL", interval="1/minute", start_date="2025-11-03", end_date="2025-12-03")
print("\nTSLA 2020-2023年季度线：")
print(df_quarter)














