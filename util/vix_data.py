
import pandas as pd
import requests

# CBOE VIX历史数据CSV直连地址
VIX_CSV_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# 下载并读取数据
def get_vix_from_cboe():
    # 发送请求获取CSV数据
    response = requests.get(VIX_CSV_URL)
    # 将数据写入本地（可选）
    with open("vix_history.csv", "wb") as f:
        f.write(response.content)
    # 用pandas解析CSV
    vix_df = pd.read_csv(VIX_CSV_URL, parse_dates=["DATE"])
    # 按日期排序
    vix_df = vix_df.sort_values(by="DATE").reset_index(drop=True)
    return vix_df

# 调用函数
vix_data = get_vix_from_cboe()
print(vix_data.tail())  # 查看最新5条数据
