# # 4. Final Panelinin Oluşturması
# 
# ## 4.1 Haftasonu Haberleri → İş Günü Kaydırma
# 
# Borsalar yalnızca iş günleri açıktır. Hafta sonu veya tatil günlerinde yayımlanan haberler, **bir sonraki iş günündeki** fiyat hareketini etkiler. Bu nedenle, takvim bazlı sentiment verisini borsa takvimine eşlenecektir.
# 
# **Stratejimiz ise:** `pd.merge_asof(direction='forward')` ile her takvim günü, ondan büyük veya eşit olan **ilk borsa gününe** eşlemek olacaktır.
# 
# ## 4.2 Finansal Değişkenleri Birleştirme
# 
# Paneli şu adımlarla oluşturduk:
# 1. Returns (1d/3d/5d/10d) geniş formattan uzun formata çevrildi
# 2. Günlük volatilite eklendi
# 3. Sentiment (trading-day bazlı) ile sol birleştirme (left join)
# 4. ESG skoru ticker bazında eklendi (zamansal boyut yok — firma-sabit)

# (FinBERT_Sentiment_Panel.xlsx) hali hazırda oluşturulmuş sentiment skorları panelinin yüklenmesi
sent = pd.read_excel(F_SENTIMENT, sheet_name="Panel Verisi")
sent["Tarih"] = pd.to_datetime(sent["Tarih"])

# Sütun isimlerinin standardizasyonu
sent = sent.rename(columns={
    "Tarih": "date",
    "Ticker": "ticker",
    "Haber Sayısı": "news_count",
    "Ort. Polarite": "polarity_mean",
    "Ort. Güven": "confidence_mean",
    "Neg. Oran": "neg_ratio",
    "Rolling Z-Score": "rolling_z",
    "Şok(Z)": "shock_z",
    "Şok(Pct)": "shock_pct",
    "Sentiment Şoku": "shock",
    "Şok Yoğunluğu": "shock_intensity",
    "Pozitif": "n_positive",
    "Negatif": "n_negative",
    "Nötr": "n_neutral",
    "Sektör": "sector_tr",
})

print(f"Sentiment panel: {sent.shape}")
print(f"Toplam şok (takvim günü dahil): {int(sent['shock'].sum())}")

# Haftasonu ve tatil günlerini kaydırma
trading_dates = pd.read_csv(F_RET_1D)["Date"]
trading_dates = pd.to_datetime(trading_dates).sort_values().reset_index(drop=True)

print(f"2023 borsa açık gün sayısı: {len(trading_dates)}")
print(f"İlk gün: {trading_dates.iloc[0].date()}   Son gün: {trading_dates.iloc[-1].date()}")

calendar_all = pd.DataFrame({
    "calendar_date": pd.date_range(sent["date"].min(), sent["date"].max(), freq="D")
})
calendar_all = pd.merge_asof(
    calendar_all.sort_values("calendar_date"),
    pd.DataFrame({"trading_date": trading_dates}).sort_values("trading_date"),
    left_on="calendar_date",
    right_on="trading_date",
    direction="forward",
)

sent = sent.merge(
    calendar_all.rename(columns={"calendar_date": "date"}),
    on="date", how="left"
)

unmatched = sent["trading_date"].isna().sum()
print(f"Eşleşmeyen takvim satırı: {unmatched} (yıl sonu geçişi — analizden çıkarılacak)")

# İşlem gününe göre genel duygu durumu seviyesi
sent_tdy = sent.dropna(subset=["trading_date"]).copy()

def weighted_mean(values, weights):
    """Haber sayısına göre ağırlıklı ortalama, sıfır ağırlıklı satırlar dikkate alınmadan hesaplanmıştır."""
    mask = (~pd.isna(values)) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])

grp = sent_tdy.groupby(["ticker", "trading_date"], as_index=False)

agg = grp.agg(
    news_count      =("news_count", "sum"),
    n_positive      =("n_positive", "sum"),
    n_negative      =("n_negative", "sum"),
    n_neutral       =("n_neutral", "sum"),
    shock           =("shock", "max"),   # shock = 1 eğer herhangi bir kaynak gününde şok yaşandıysa
    shock_intensity =("shock_intensity", "max"),
)

wavg = sent_tdy.groupby(["ticker", "trading_date"]).apply(
    lambda g: pd.Series({
        "polarity_mean":   weighted_mean(g["polarity_mean"].values, g["news_count"].values),
        "confidence_mean": weighted_mean(g["confidence_mean"].values, g["news_count"].values),
        "neg_ratio":       weighted_mean(g["neg_ratio"].values, g["news_count"].values),
        "rolling_z":       weighted_mean(g["rolling_z"].values, g["news_count"].values),
    }),
    include_groups=False,
).reset_index()

sent_daily = agg.merge(wavg, on=["ticker", "trading_date"]).rename(columns={"trading_date": "date"})
print(f"Trading-day sentiment paneli: {sent_daily.shape}")
print(f"İş günü bazlı toplam şok: {int(sent_daily['shock'].sum())}")

# Geniş formatlı getirileri uzun formata dönüştürme ve son birleştirme.
def wide_to_long(filepath, value_name):
    """Pivot wide return/volatility tables to (date, ticker, value) format."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.melt(id_vars="date", var_name="ticker", value_name=value_name)
    df["ticker"] = df["ticker"].str.strip()
    return df.dropna(subset=[value_name])

ret_1d  = wide_to_long(F_RET_1D,  "return_1d")
ret_3d  = wide_to_long(F_RET_3D,  "return_3d")
ret_5d  = wide_to_long(F_RET_5D,  "return_5d")
ret_10d = wide_to_long(F_RET_10D, "return_10d")
vol_d   = wide_to_long(F_VOL,     "volatility")

# Tüm finansal değişkenleri birleştirme
financial = ret_1d
for df_next in [ret_3d, ret_5d, ret_10d, vol_d]:
    financial = financial.merge(df_next, on=["date", "ticker"], how="outer")

print(f"Finansal panel: {financial.shape}")

# ESG Risk Skorlarının yüklenmesi
esg = pd.read_csv(F_ESG, sep=';', header=1, on_bad_lines='skip')
print(esg.columns)
esg = esg.rename(columns={
    "Symbol": "ticker",
    "Name": "company_name",
    "Sector": "sector_en",
    "Total ESG Risk Score": "esg_total",
    "Environment Risk Score": "esg_e",
    "Governance Risk Score": "esg_g",
    "Social Risk Score": "esg_s",
    "Controversy Level": "controversy_level",
    "Controversy Score": "controversy_score",
    "ESG Risk Percentile": "esg_percentile",
    "ESG Risk Level": "esg_level",
})

sector_map_fr = {
    "Technology": "Technologie",
    "Financial Services": "Finance",
    "Healthcare": "Santé",
    "Energy": "Énergie",
}
esg["sector_fr"] = esg["sector_en"].map(sector_map_fr)

esg_cols = ["ticker", "company_name", "sector_en", "sector_fr",
            "esg_total", "esg_e", "esg_s", "esg_g",
            "controversy_level", "controversy_score", "esg_percentile", "esg_level"]
esg = esg[esg_cols]

print(f"ESG tablosu: {esg.shape}")
esg[["ticker", "sector_en", "esg_total", "esg_e", "esg_s", "esg_g", "esg_level"]]

# Final Panelinin oluşturulması: financial ← sentiment (solda) ← ESG (solda)
panel = financial.merge(sent_daily, on=["date", "ticker"], how="left")

# Eksik şok göstergelerinin 0 ile doldurulması (haber yok, şok da yok).
zero_fill_cols = ["shock", "news_count", "n_positive", "n_negative",
                  "n_neutral", "shock_intensity"]
for c in zero_fill_cols:
    if c in panel.columns:
        panel[c] = panel[c].fillna(0)

# ESG eklnmesi (firm-level, time-invariant)
panel = panel.merge(esg, on="ticker", how="left")
panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

# Panel ID
panel["firm_id"] = panel["ticker"].astype("category").cat.codes
panel["t_id"]    = panel["date"].rank(method="dense").astype(int)

print(f"\n{'='*60}")
print(f"FINAL PANEL: {panel.shape[0]:,} satır × {panel.shape[1]} sütun")
print(f"Beklenen: 250 iş günü × 12 şirket = 3,000 satır")
print(f"\nKalite Kontrolleri:")
days_per_firm = panel.groupby('ticker')['date'].nunique()
print(f"  Dengeli panel: {days_per_firm.nunique() == 1}")
print(f"  Toplam şok (iş günü): {int(panel['shock'].sum())}")
print(f"\nEksik değerler (ana değişkenler):")
key_cols = ["return_1d", "return_3d", "return_5d", "return_10d",
            "volatility", "shock", "esg_total", "esg_e", "esg_s", "esg_g"]
print(panel[key_cols].isna().sum().to_string())

with pd.ExcelWriter(OUT_PANEL, engine="openpyxl") as w:
    panel.to_excel(w, sheet_name="Final Panel", index=False)
    esg.to_excel(w, sheet_name="ESG Scores", index=False)
print(f"\n✓ Final panel kaydedildi: {OUT_PANEL}")

