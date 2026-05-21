# # 3. Veri Hazırlığı / Préparation des Données
# 
# ## 3.1 Haber Verisi Yükleme
# 
# Ham veri seti `verisetbirles_ikson.xlsx` dosyasında yer almaktadır. Bu dosya üç farklı kaynaktan birleştirilmiştir:
# - **AlphaVantage** — finansal haber API'si
# - **Marketaux** — ticker bazlı haber verisi  
# - **Kaggle (Financial News with Sentiment)** — etiketlenmiş finansal haberler
# 
# Sadece `news_type == 'company'` ve geçerli `ticker` eşleşmesi olan haberler analize dahil edilmektedir.

# Haber Verilerinin yüklenmesi
def load_data(input_file):
    df = pd.read_excel(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # Ticker bazlı eleme
    company_df = df[
        df['ticker'].notna() & (df['news_type'] == 'company')
    ].copy()
    company_df = company_df.sort_values(['ticker', 'date']).reset_index(drop=True)

    print(f"  Toplam haber : {len(company_df):,}")
    print(f"  Şirket sayısı: {company_df['ticker'].nunique()}")
    print(f"  Dönem        : {company_df['date'].min().date()} → {company_df['date'].max().date()}")
    return company_df

company_df = load_data(F_RAW_NEWS)
company_df[['date', 'ticker', 'title', 'source']].head(5)

# ## 3.2 FinBERT ile Sentiment Analizi
# 
# **FinBERT** (ProsusAI/finbert), finansal metinler üzerinde fine-tune edilmiş bir BERT modelidir. Genel amaçlı dil modellerine kıyasla finansal terminolojiyi çok daha iyi kavrar ve bu sebeple seçilmiştir.
# 
# Her haber için model şu çıktıları üretecektir:
# - **Label**: `positive` / `negative` / `neutral`
# - **Confidence score**: Modelin tahminine olan güveni (0–1 arası)
# 
# **Polarite Skoru** aşağıdaki formülle hesaplanmıştır:
# 
# $$\text{polarity}_{it} = \begin{cases} +\text{confidence} & \text{eğer pozitif} \\ -\text{confidence} & \text{eğer negatif} \\ 0 & \text{eğer nötr} \end{cases}$$

# FinBERT pipeline konfigürasyonu
CACHE_FILE           = "finbert_cache.csv"
ROLLING_WINDOW       = 30    # days for rolling z-score
ZSCORE_THRESHOLD     = -1.5  # shock detection threshold
PERCENTILE_THRESHOLD = 10    # bottom percentile cutoff
BATCH_SIZE           = 32    # GPU batch size
USE_FINBERT          = True  # set False to use lexicon fallback

# Loughran-McDonald financial lexicon
POSITIVE_WORDS = {
    'record', 'beat', 'beats', 'exceeded', 'growth', 'profit', 'profitable',
    'revenue', 'gains', 'surge', 'surged', 'rally', 'strong', 'upgrade',
    'upgraded', 'outperform', 'bullish', 'expansion', 'breakthrough',
    'dividend', 'buyback', 'approved', 'launch', 'increase', 'improvement',
    'momentum', 'milestone', 'success', 'robust', 'solid', 'win', 'recovery'
}
NEGATIVE_WORDS = {
    'loss', 'losses', 'decline', 'declined', 'decrease', 'miss', 'missed',
    'shortfall', 'deficit', 'debt', 'lawsuit', 'investigation', 'fine',
    'layoff', 'restructuring', 'bankruptcy', 'default', 'downgrade',
    'concern', 'uncertainty', 'slump', 'crash', 'fraud', 'scandal',
    'warning', 'disappointing', 'weak', 'challenging', 'failed', 'crisis'
}

print("✓ FinBERT konfigürasyonu ayarlandı.")
print(f"  Rolling pencere : {ROLLING_WINDOW} gün")
print(f"  Z-score eşiği   : {ZSCORE_THRESHOLD}")
print(f"  Percentile      : Alt {PERCENTILE_THRESHOLD}. yüzdelik")

def run_finbert(texts, batch_size=BATCH_SIZE):
    """Run ProsusAI/finbert on a list of text strings."""
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print(f"  Model yükleniyor... ({'GPU' if device == 0 else 'CPU'})")

    clf = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=False,
        device=device,
        truncation=True,
        max_length=512
    )

    results = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t)[:512] for t in texts[i:i + batch_size]]
        out = clf(batch)
        for o in out:
            results.append({'label': o['label'].lower(), 'score': round(float(o['score']), 4)})
        if (i + batch_size) % 1000 == 0:
            print(f"  {min(i + batch_size, len(texts)):,} / {len(texts):,} haber işlendi")
    return results


def run_lexikon(texts):
    """Loughran-McDonald sözlük tabanlı duygu puanlayıcısı"""
    results = []
    for text in texts:
        words = str(text).lower().split()
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            results.append({'label': 'neutral', 'score': 0.60})
        else:
            conf = round(0.55 + 0.40 * (max(pos, neg) / total), 4)
            if pos > neg:
                results.append({'label': 'positive', 'score': conf})
            elif neg > pos:
                results.append({'label': 'negative', 'score': conf})
            else:
                results.append({'label': 'neutral', 'score': 0.60})
    return results


def hesapla_sentiment(company_df):
    print("\nFinBERT sentiment analizi başlıyor...")

    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE)
        if len(cache) == len(company_df):
            print(f"  Cache bulundu → yeniden hesaplamadan yükleniyor ({CACHE_FILE})")
            company_df['finbert_label'] = cache['finbert_label'].values
            company_df['finbert_score'] = cache['finbert_score'].values
            print(f"  Dağılım: {company_df['finbert_label'].value_counts().to_dict()}")
            return company_df
        else:
            print("  Cache boyutu uyuşmuyor ")

    texts = (company_df['text'].fillna('') + ' ' + company_df['title'].fillna('')).tolist()

    if USE_FINBERT:
        try:
            results = run_finbert(texts)
        except Exception as e:
            print(f"  FinBERT yüklenemedi: {e}\n  → Lexikon yöntemine geçiliyor...")
            results = run_lexikon(texts)
    else:
        results = run_lexikon(texts)

    company_df['finbert_label'] = [r['label'] for r in results]
    company_df['finbert_score'] = [r['score'] for r in results]
    company_df[['finbert_label', 'finbert_score']].to_csv(CACHE_FILE, index=False)
    print(f"  Cache kaydedildi: {CACHE_FILE}")
    print(f"  Dağılım: {company_df['finbert_label'].value_counts().to_dict()}")
    return company_df


company_df = hesapla_sentiment(company_df)

# ## 3.3 Günlük Agregasyon ve Şok Tespiti
# 
# ### Günlük Agregasyon
# Her `(ticker, gün)` çifti için birden fazla haber bulunabilir bu sebeple Haberleri günlük düzeye indirmek gereklidir:
# - **Ortalama polarite** hesaplanır (tüm haber polaritelerinin ortalaması)
# - **Negatif oran** hesaplanır: $\text{neg\_ratio} = n_{\text{neg}} / n_{\text{total}}$
# 
# ### Şok Tespiti: İki Koşullu Yaklaşım
# 
# Bir günün negatif sentiment şoku olarak etiketlenmesi için **her iki koşulun aynı anda** sağlanması gerekmektedir:
# 
# **Koşul 1 — Z-Score Eşiği:**
# $$z_{it} = \frac{\bar{p}_{it} - \mu_{i,30g}}{\sigma_{i,30g} + \epsilon} < -1.5$$
# 
# **Koşul 2 — Percentile Eşiği:**
# $$\bar{p}_{it} < P_{10}(\text{ticker}_i)$$
# 
# Bu çift-koşullu yaklaşım, yalnızca z-score ya da yalnızca percentile kullanmaya kıyasla daha güvenilir ve anlamlı şok tespiti sağlayacaktır.

def gunluk_agregasyon(company_df):

    def polarite(row):
        if row['finbert_label'] == 'positive':
            return row['finbert_score']
        elif row['finbert_label'] == 'negative':
            return -row['finbert_score']
        return 0.0

    company_df['polarity_score'] = company_df.apply(polarite, axis=1)

    daily = company_df.groupby(['ticker', 'date']).agg(
        n_news       =('polarity_score', 'count'),
        avg_polarity =('polarity_score', 'mean'),
        n_positive   =('finbert_label', lambda x: (x == 'positive').sum()),
        n_negative   =('finbert_label', lambda x: (x == 'negative').sum()),
        n_neutral    =('finbert_label', lambda x: (x == 'neutral').sum()),
        avg_conf     =('finbert_score', 'mean'),
    ).reset_index()

    daily['neg_ratio'] = daily['n_negative'] / daily['n_news']

    # Sector lookup
    if 'sector' in company_df.columns:
        sector_map = company_df.groupby('ticker')['sector'].first()
        daily['sector'] = daily['ticker'].map(sector_map)

    print(f"  Günlük panel: {daily.shape} — {daily['ticker'].nunique()} firma")
    return company_df, daily


def sok_tespiti(daily):
    epsilon = 1e-8

    daily = daily.sort_values(['ticker', 'date']).copy()

    # Koşul 1 : Rolling z-score
    daily['rolling_mean'] = (
        daily.groupby('ticker')['avg_polarity']
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=5).mean())
    )
    daily['rolling_std'] = (
        daily.groupby('ticker')['avg_polarity']
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=5).std())
    )
    daily['rolling_zscore'] = (
        (daily['avg_polarity'] - daily['rolling_mean']) /
        (daily['rolling_std'] + epsilon)
    )
    daily['shock_zscore'] = (daily['rolling_zscore'] < ZSCORE_THRESHOLD).astype(int)

    # Koşul 2: Firm-specific percentile
    pct_threshold = daily.groupby('ticker')['avg_polarity'].transform(
        lambda x: x.quantile(PERCENTILE_THRESHOLD / 100)
    )
    daily['shock_pct'] = (daily['avg_polarity'] < pct_threshold).astype(int)

    # Combined shock: 2 koşulu sağlayan şoklar
    daily['sentiment_shock'] = (
        (daily['shock_zscore'] == 1) & (daily['shock_pct'] == 1)
    ).astype(int)

    # Şok yoğunluğu = |z-score| 'u şok günlerinde
    daily['shock_intensity'] = np.where(
        daily['sentiment_shock'] == 1,
        daily['rolling_zscore'].abs(),
        0.0
    )

    n_shocks = daily['sentiment_shock'].sum()
    pct = n_shocks / len(daily) * 100
    print(f"  Toplam şok: {n_shocks} ({pct:.2f}% gözlem)")
    print(f"  Firma başına şok dağılımı:")
    print(daily.groupby('ticker')['sentiment_shock'].sum().to_string())
    return daily


company_df, daily = gunluk_agregasyon(company_df)
daily = sok_tespiti(daily)

