# 
# # FİNANSAL GETİRİ VE VOLATİLİTE VERİLERİNİN HAZIRLIĞI
# 
# Colab ortamında yfinance kütüphanesi kullanılarak 1 -3 -5 -10 günlük finansal getirilerin ve volatilitenin çekilmesi

import yfinance as yf
import pandas as pd
import numpy as np
from google.colab import files

#Şirket Listesi
tickers = ['MCHP', 'NVDA', 'VRSN', 'COP', 'KMI', 'OKE', 'LLY', 'UNH', 'UHS', 'V', 'GS', 'WFC']

print("Veriler Yahoo Finance üzerinden çekiliyor...")
raw_data = yf.download(tickers, start="2022-12-15", end="2023-12-31")['Close']

#Getiri hesabı
prices = raw_data.loc['2023-01-01':]
ret1 = raw_data.pct_change(1).loc['2023-01-01':]
ret3 = raw_data.pct_change(3).loc['2023-01-01':]
ret5 = raw_data.pct_change(5).loc['2023-01-01':]
ret10 = raw_data.pct_change(10).loc['2023-01-01':]
vol = raw_data.pct_change(1).rolling(window=10).std().loc['2023-01-01':]

#Panel formatına hazır hale getirme
def make_long(df, name):
    return df.stack().reset_index().rename(columns={'level_1': 'Ticker', 0: name, 'Date': 'Tarih'})

df_prices = make_long(prices, 'Price')
df_ret1 = make_long(ret1, 'Return_1d')
df_ret3 = make_long(ret3, 'Return_3d')
df_ret5 = make_long(ret5, 'Return_5d')
df_ret10 = make_long(ret10, 'Return_10d')
df_vol = make_long(vol, 'Volatility_Daily')

final_financials = df_prices.merge(df_ret1, on=['Tarih', 'Ticker']) \
                             .merge(df_ret3, on=['Tarih', 'Ticker']) \
                             .merge(df_ret5, on=['Tarih', 'Ticker']) \
                             .merge(df_ret10, on=['Tarih', 'Ticker']) \
                             .merge(df_vol, on=['Tarih', 'Ticker'])

final_financials = final_financials[['Tarih', 'Ticker', 'Price', 'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d', 'Volatility_Daily']]

metrics = {
    'PRICES_2023.csv': prices,
    'RETURN_1D_2023.csv': ret1,
    'RETURN_3D_2023.csv': ret3,
    'RETURN_5D_2023.csv': ret5,
    'RETURN_10D_2023.csv': ret10,
    'VOLATILITY_DAILY_2023.csv': vol
}

for file_name, df in metrics.items():
    df.to_csv(file_name)
    files.download(file_name)


DATA_DIR = Path("/content")

# Input files
F_RAW_NEWS  = DATA_DIR / "verisetbirles_ikson.xlsx"
F_SENTIMENT = DATA_DIR / "FinBERT_Sentiment_Panel.xlsx"
F_RET_1D    = DATA_DIR / "RETURN_1D_2023.csv"
F_RET_3D    = DATA_DIR / "RETURN_3D_2023.csv"
F_RET_5D    = DATA_DIR / "RETURN_5D_2023.csv"
F_RET_10D   = DATA_DIR / "RETURN_10D_2023.csv"
F_VOL       = DATA_DIR / "VOLATILITY_DAILY_2023.csv"
F_ESG       = DATA_DIR / "/content/esgsriskcores2023.csv"
F_PANEL_VIZ = DATA_DIR / "final_panel_2023_formatted (2).xlsx"

# Output files
OUT_FINBERT = DATA_DIR / "FinBERT_Sentiment_Panel_2023.xlsx"
OUT_PANEL   = DATA_DIR / "final_panel_2023.xlsx"
OUT_RESULTS = DATA_DIR / "regression_results.xlsx"

os.makedirs('figures', exist_ok=True)

print("Dosya varlık kontrolü:")
all_inputs = [F_RAW_NEWS, F_SENTIMENT, F_RET_1D, F_RET_3D, F_RET_5D,
              F_RET_10D, F_VOL, F_ESG, F_PANEL_VIZ]
for f in all_inputs:
    status = "yüklenmiş" if f.exists() else "EKSİK"
    print(f"  {status}  {f.name}")

