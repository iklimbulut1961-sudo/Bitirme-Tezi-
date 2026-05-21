# # 8. Sonuç ve Değerlendirme / Conclusion
# 
# ## 8.1 Bulgular
# 
# Tezimizde, **2023 yılında 12 S&P 500 şirketine ait 11.877 haber** üzerinde FinBERT tabanlı sentiment analizi gerçekleştirmiş ve toplam **131 negatif sentiment şoku** tespit etmiştir (%4.37).
# 
# İki-yönlü sabit etkiler panel regresyonun bulguları:
# 
# **H1 — Şokun Doğrudan Etkisi (β₁):**  
# Negatif sentiment şokları, hisse senedi getirileri üzerinde negatif ve istatistiksel olarak anlamlı bir etki göstermiştir. Bu bulgu, piyasaların negatif haberlere anında tepki verdiğini doğrulamaktadır.
# 
# **H2 — ESG Moderatör Etkisi (β₂):**
# 
# | Hipotez | Sonuç | Yorum |
# |---|---|---|
# | H2 (ESG Total × Return) | **DESTEKLENDİ** (p=0.006) | Yüksek ESG riski şok etkisini ağırlaştırır |
# | H2a (ESG_E × Return) | Kısmi destek (p=0.087) | Çevresel risk marjinal etkiye sahip |
# | H2b (ESG_S × Return) | Reddedildi | Sosyal risk getiriyi modere etmez |
# | H2c (ESG_G × Return) | Reddedildi | Yönetişim riski getiriyi modere etmez |
# | H2 (ESG_S × Volatility) | **DESTEKLENDİ** (p=0.002) | Sosyal risk volatiliteyi önemli ölçüde artırır |
# 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# Tablo 5.5 (Bölüm 5.3.1) sonuçları:
#   M5: Volatilite × ESG_Total  → β1=-0.00014, β2=+0.00032, p=0.618 (ns)
#   M7: Volatilite × ESG_S      → β1=-0.00002, β2=+0.00135, p=0.002 ***
#
# Portföy optimizasyonunda "şok anındaki marjinal volatilite etkisi"ni
# modellemek için M7 katsayılarını (en anlamlı volatilite modeli) kullanıyoruz.
# M1 katsayıları getiri ayarlaması için kullanılır.

BETA = {
    # Volatilite modeli (M7: ESG_S × Volatilite) — şok etkisi
    "beta1_vol"    : -0.00002,   # şokun doğrudan volatilite etkisi
    "beta2_vol_S"  : +0.00135,   # ESG_S moderasyon katsayısı (p<0.01) ***

    # Getiri modeli (M1: ESG_Total × Return_1d) — şok etkisi
    "beta1_ret"    : -0.00080,   # şokun doğrudan getiri etkisi
    "beta2_ret_T"  : +0.00217,   # ESG_Total moderasyon katsayısı (p<0.01) ***
}

print("TWFE katsayıları yüklendi:")
for k, v in BETA.items():
    print(f"  {k:20s} = {v:+.5f}")

# HÜCRE 2 — Firma verileri
# =====================================================================
# ESG skorları (Tablo 4.1), volatilite ortalamaları (Tablo 5.1/5.3),
# şok günü getiri deltas (Tablo 5.7) — tüm değerler tezden alındı.

firms = pd.DataFrame({
    "ticker"   : ["NVDA","VRSN","MCHP","V","GS","WFC","UNH","LLY","UHS","KMI","OKE","COP"],
    "sector"   : ["Tech","Tech","Tech","Finance","Finance","Finance",
                  "Health","Health","Health","Energy","Energy","Energy"],
    "esg_total": [13.6, 21.3, 31.8,  16.7, 25.5, 36.2,  15.3, 24.3, 33.0,  19.0, 25.1, 33.9],
    "esg_e"    : [ 2.3,  4.7, 15.3,   1.8,  0.9,  2.0,   0.0,  2.5,  3.7,   8.3, 10.8, 17.6],
    "esg_s"    : [ 4.9, 10.8,  9.2,   8.2, 12.8, 14.8,   9.7, 12.6, 20.7,   7.2, 11.0,  8.1],
    "esg_g"    : [ 6.3,  5.8,  7.4,   6.7, 11.8, 19.4,   5.6,  9.1,  8.6,   3.5,  3.3,  8.3],
    # Aylık ortalama volatilite (Tablo 5.3 ve heatmap verilerinden)
    "vol_base" : [0.0203, 0.0118, 0.0208, 0.0121, 0.0144, 0.0133,
                  0.0135, 0.0156, 0.0154, 0.0109, 0.0143, 0.0179],
    # Yıllık ortalama getiri (Tablo 5.1'den ölçeklendirilmiş)
    "ret_mean" : [0.0015, 0.0009, 0.0008, 0.0012, 0.0011, 0.0007,
                  0.0016, 0.0018, 0.0007, 0.0010, 0.0008, 0.0011],
    "esg_level": ["Low","Medium","High","Low","Medium","High",
                  "Low","Medium","High","Low","Medium","High"],
})

# ESG skorlarını standardize et (z-score) — regresyonda olduğu gibi
for col in ["esg_total","esg_e","esg_s","esg_g"]:
    mu = firms[col].mean()
    sd = firms[col].std()
    firms[f"{col}_z"] = (firms[col] - mu) / sd

print("\nFirma verileri:")
print(firms[["ticker","sector","esg_level","esg_total","esg_s","vol_base"]].to_string(index=False))
N = len(firms)



# =====================================================================
# HÜCRE 3 — Kovaryans matrisi oluştur
# =====================================================================
# Günlük log-return verilerinden hesapla. Eğer Colab'da final_panel
# mevcutsa oradan okur; yoksa tarihsel volatiliteden köşegen matris kurar.

DATA_DIR = Path("/content")
F_PANEL  = DATA_DIR / "final_panel_2023_formatted_2.xlsx"

if F_PANEL.exists():
    # Excel'de ilk 4 satır başlık/meta bilgi → header=4
    panel = pd.read_excel(F_PANEL, sheet_name="Panel Verisi", header=4)

    # Kolon adlarındaki newline karakterlerini temizle
    panel.columns = [str(c).replace("\n", " ").strip() for c in panel.columns]

    # Tarih ve ticker kolonlarını normalize et
    panel = panel.rename(columns={
        "Tarih"    : "date",
        "Ticker"   : "ticker",
        "Return 1d": "return_1d",
    })
    # Tarih "MM.DD.YYYY" string formatında geliyor → format='mixed' ile parse et
    panel["date"] = pd.to_datetime(panel["date"], format="mixed")

    # Sadece gerekli kolonları al, NaN satırları çıkar
    panel = panel[["date", "ticker", "return_1d"]].dropna()

    # Geniş format: satır=tarih, sütun=ticker
    ret_wide = panel.pivot(index="date", columns="ticker", values="return_1d")

    # Sıralamayı firms DataFrame ile hizala
    tickers_ordered = firms["ticker"].tolist()
    ret_wide = ret_wide.reindex(columns=tickers_ordered).dropna()

    COV_MATRIX = ret_wide.cov().values
    print(f"✓ Kovaryans matrisi veri setinden hesaplandı ({ret_wide.shape[0]} gün, {ret_wide.shape[1]} firma)")
else:
    # Yedek: köşegen matris (korelasyonlar ihmal)
    vol = firms["vol_base"].values
    COV_MATRIX = np.diag(vol**2)
    print("⚠ Panel verisi bulunamadı — köşegen kovaryans matrisi kullanılıyor.")
    print("  Colab'a final_panel_2023_formatted_2.xlsx yükleyiniz.")

print(f"Kovaryans matrisi boyutu: {COV_MATRIX.shape}")



# =====================================================================
# HÜCRE 4 — Şok döneminde efektif volatilite ayarlaması
# =====================================================================
# Temel fikir: Şok geldiğinde her firmanın volatilitesi TWFE katsayılarına
# göre farklı etkileniyor. ESG_S yüksek firmalar daha fazla etkileniyor (M7).
#
#   σ_shock(i) = σ_base(i) + |β1_vol| + β2_vol_S × ESG_S_z(i)
#
# Bu "şok anındaki marginal volatilite artışı" portföy optimizasyonunda
# bir risk katmanı olarak kullanılır.

def compute_shock_volatility(firms_df, beta1, beta2_s):
    """Şok döneminde her firma için efektif volatilite hesapla."""
    shock_addon = abs(beta1) + beta2_s * firms_df["esg_s_z"].values
    return firms_df["vol_base"].values + shock_addon

# Her firmanın şok dönemindeki efektif volatilitesi
firms["vol_shock"] = compute_shock_volatility(
    firms,
    BETA["beta1_vol"],
    BETA["beta2_vol_S"]
)

# Şok anındaki getiri beklentisi (M1'den getiri moderasyonu)
firms["ret_shock"] = (firms["ret_mean"].values
                      + BETA["beta1_ret"]
                      + BETA["beta2_ret_T"] * firms["esg_total_z"].values)

print("\nFirma bazında volatilite ve getiri (normal vs şok):")
print(firms[["ticker","esg_level","vol_base","vol_shock","ret_mean","ret_shock"]
            ].round(5).to_string(index=False))


# HÜCRE 5 — Portföy optimizasyon fonksiyonları
# =====================================================================

def portfolio_variance(weights, cov_matrix):
    """Portföy varyansı: w' Σ w"""
    return weights @ cov_matrix @ weights

def portfolio_return(weights, expected_returns):
    """Portföy beklenen getirisi: w' μ"""
    return weights @ expected_returns

def min_variance_portfolio(cov_matrix, expected_returns,
                           target_return=None, allow_short=False):
    """
    Minimum varyans portföyü bul.

    Kısıtlar:
      1. Σ wi = 1  (tam yatırım)
      2. wi ≥ 0    (açığa satış yasak — unless allow_short=True)
      3. Eğer target_return verilmişse: w'μ ≥ target_return

    Döndürür: optimize sonucu (weights, variance, return)
    """
    n = len(expected_returns)
    w0 = np.ones(n) / n  # başlangıç: eşit ağırlık

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: portfolio_return(w, expected_returns) - target_return
        })
    bounds = None if allow_short else [(0, 1)] * n

    result = minimize(
        portfolio_variance,
        w0,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )
    return result


def build_shock_cov(cov_base, shock_vol_adj):
    """
    Şok kovaryans matrisini oluştur.
    Korelasyon yapısını koruyarak volatiliteleri güncelle.
    D_shock = diag(σ_shock / σ_base)  → Σ_shock = D_shock × Σ_base × D_shock
    """
    ratio = shock_vol_adj / np.sqrt(np.diag(cov_base))
    D = np.diag(ratio)
    return D @ cov_base @ D



# =====================================================================
# HÜCRE 6 — Üç senaryo için optimizasyon
# =====================================================================
# Senaryo 1: Baseline — normal piyasa koşulları
# Senaryo 2: Şok (ESG ayarlaması YOK) — tüm firmalar eşit şok alır
# Senaryo 3: ESG-Ayarlı Şok — TWFE katsayıları uygulanır

# Kovaryans matrisleri
COV_BASELINE = COV_MATRIX.copy()

# Şok senaryosu (uniform %20 volatilite artışı — ESG etkisi yok)
shock_uniform = firms["vol_base"].values * 1.20
COV_SHOCK_UNIFORM = build_shock_cov(COV_BASELINE, shock_uniform)

# ESG-ayarlı şok (TWFE sonuçları uygulandı)
COV_SHOCK_ESG = build_shock_cov(COV_BASELINE, firms["vol_shock"].values)

# Beklenen getiriler
MU_BASE  = firms["ret_mean"].values
MU_SHOCK = firms["ret_shock"].values

# Optimizasyon — 3 senaryo
print("=" * 60)
print("OPTİMİZASYON SONUÇLARI")
print("=" * 60)

scenarios = [
    ("Baseline (Normal Piyasa)",        COV_BASELINE,     MU_BASE,  "baseline"),
    ("Şok (ESG ayarsız, uniform +20%)", COV_SHOCK_UNIFORM,MU_SHOCK, "shock_uniform"),
    ("ESG-Ayarlı Şok (TWFE)",          COV_SHOCK_ESG,    MU_SHOCK, "shock_esg"),
]

opt_results = {}
for name, cov, mu, key in scenarios:
    res = min_variance_portfolio(cov, mu)
    w   = res.x
    pvar = portfolio_variance(w, cov)
    pret = portfolio_return(w, mu)
    opt_results[key] = {
        "name"     : name,
        "weights"  : w,
        "variance" : pvar,
        "std"      : np.sqrt(pvar),
        "return"   : pret,
        "sharpe"   : pret / np.sqrt(pvar) if pvar > 0 else 0,
    }

    print(f"\n{name}")
    print(f"  Portföy Std Dev  : {np.sqrt(pvar)*100:.4f}%")
    print(f"  Portföy Getiri   : {pret*100:.4f}%")
    print(f"  Sharpe (approx)  : {pret/np.sqrt(pvar):.3f}")
    print(f"  Ağırlıklar:")
    for i, (t, lv) in enumerate(zip(firms["ticker"], firms["esg_level"])):
        if w[i] > 0.001:
            print(f"    {t:6s} ({lv:6s}) : {w[i]*100:6.2f}%")

# =====================================================================
# HÜCRE 7 — Ağırlık değişim tablosu (Baseline → ESG Şok)
# =====================================================================
w_base = opt_results["baseline"]["weights"]
w_esg  = opt_results["shock_esg"]["weights"]

weight_df = firms[["ticker","sector","esg_level","esg_total","esg_s"]].copy()
weight_df["w_baseline (%)"]  = (w_base * 100).round(2)
weight_df["w_shock_esg (%)"] = (w_esg  * 100).round(2)
weight_df["Δw (pp)"]         = (weight_df["w_shock_esg (%)"] - weight_df["w_baseline (%)"]).round(2)

print("\n" + "=" * 80)
print("AĞIRLIK DEĞİŞİMİ: Baseline → ESG-Ayarlı Şok Senaryosu")
print("=" * 80)
print(weight_df.to_string(index=False))

# Temel bulgu: ESG High firmalar şok döneminde ağırlık kazanıyor mu?
high_delta = weight_df[weight_df["esg_level"]=="High"]["Δw (pp)"].mean()
low_delta  = weight_df[weight_df["esg_level"]=="Low"]["Δw (pp)"].mean()
print(f"\nESG High firmaların ortalama ağırlık değişimi : {high_delta:+.2f} pp")
print(f"ESG Low  firmaların ortalama ağırlık değişimi : {low_delta:+.2f} pp")


# =====================================================================
# HÜCRE 8 — Verimli Sınır (Efficient Frontier) Grafiği
# =====================================================================
def efficient_frontier(cov, mu, n_points=100):
    """Farklı hedef getirilerde min-variance portföyleri hesapla."""
    ret_min = mu.min() * 0.5
    ret_max = mu.max() * 1.5
    targets = np.linspace(ret_min, ret_max, n_points)

    frontier_vols  = []
    frontier_rets  = []

    for r_target in targets:
        res = min_variance_portfolio(cov, mu, target_return=r_target)
        if res.success:
            frontier_vols.append(np.sqrt(portfolio_variance(res.x, cov)) * 100)
            frontier_rets.append(portfolio_return(res.x, mu) * 100)

    return np.array(frontier_vols), np.array(frontier_rets)


fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "Optimisation de Portefeuille ESG — Frontière Efficiente\n"
    "Scénarios : Normal vs Choc ESG-Ajusté (TWFE β₂)",
    fontsize=13, fontweight="bold", y=1.01
)

# ---- Grafik 1: Verimli Sınır Karşılaştırması ----
ax1 = axes[0]
colors_ef = {
    "baseline"     : ("#2196F3", "Baseline (marché normal)"),
    "shock_uniform": ("#FF9800", "Choc uniforme (+20%)"),
    "shock_esg"    : ("#F44336", "Choc ESG-ajusté (TWFE)"),
}

for key, cov_key, mu_key in [
    ("baseline",      COV_BASELINE,     MU_BASE),
    ("shock_uniform", COV_SHOCK_UNIFORM, MU_SHOCK),
    ("shock_esg",     COV_SHOCK_ESG,    MU_SHOCK),
]:
    vols, rets = efficient_frontier(cov_key, mu_key)
    clr, lbl = colors_ef[key]
    ax1.plot(vols, rets, color=clr, linewidth=2.5, label=lbl)

    # Optimal portföy noktası
    opt = opt_results[key]
    ax1.scatter(opt["std"]*100, opt["return"]*100,
                color=clr, s=120, zorder=5, edgecolors="white", linewidth=1.5)

ax1.set_xlabel("Volatilité du portefeuille (%/jour)", fontsize=11)
ax1.set_ylabel("Rendement espéré (%/jour)", fontsize=11)
ax1.set_title("Frontière efficiente — 3 scénarios", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Bireysel firmalar
for i, row in firms.iterrows():
    clr_f = {"Low":"#4CAF50","Medium":"#FF9800","High":"#F44336"}[row["esg_level"]]
    ax1.scatter(row["vol_base"]*100, row["ret_mean"]*100,
                color=clr_f, s=60, alpha=0.7, zorder=4)
    ax1.annotate(row["ticker"], (row["vol_base"]*100, row["ret_mean"]*100),
                 textcoords="offset points", xytext=(4, 2), fontsize=7, alpha=0.8)
    # Lejant: ESG seviyeleri
patches_esg = [
    mpatches.Patch(color="#4CAF50", label="ESG Low risk"),
    mpatches.Patch(color="#FF9800", label="ESG Medium risk"),
    mpatches.Patch(color="#F44336", label="ESG High risk"),
]
ax1.legend(handles=ax1.lines + patches_esg,
           labels=[colors_ef[k][1] for k in colors_ef] +
                  ["Firme ESG Low","Firme ESG Medium","Firme ESG High"],
           fontsize=8, loc="lower right")

# ---- Grafik 2: Ağırlık Değişimi ----
ax2 = axes[1]
x = np.arange(N)
width = 0.35

bars_base = ax2.bar(x - width/2, w_base * 100, width,
                    label="Baseline", color="#2196F3", alpha=0.8)
bars_esg  = ax2.bar(x + width/2, w_esg  * 100, width,
                    label="ESG-Ajusté Choc", color="#F44336", alpha=0.8)

# ESG seviyesine göre arka plan rengi
esg_color_map = {"Low":"#E8F5E9","Medium":"#FFF3E0","High":"#FFEBEE"}
for i, row in firms.iterrows():
    ax2.axvspan(i - 0.5, i + 0.5, alpha=0.15,
                color=esg_color_map[row["esg_level"]], zorder=0)

ax2.set_xticks(x)
ax2.set_xticklabels(
    [f"{t}\n({lv[0]})" for t, lv in zip(firms["ticker"], firms["esg_level"])],
    fontsize=9
)
ax2.set_ylabel("Poids optimal (%)", fontsize=11)
ax2.set_title("Poids optimaux : Baseline vs Choc ESG-ajusté\n"
              "(Fond : Vert=Low, Orange=Medium, Rouge=High ESG risk)",
              fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

OUT_FIG = DATA_DIR / "portfolio_optimization_figure.png"
plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓ Grafik kaydedildi: {OUT_FIG}")


# =====================================================================
# HÜCRE 9 — ESG seviyesine göre portföy performans özeti
# =====================================================================
# Her ESG grubundan oluşan eşit ağırlıklı portföy karşılaştırması

esg_groups = ["Low", "Medium", "High"]
group_results = []

for grp in esg_groups:
    idx = firms[firms["esg_level"] == grp].index.tolist()
    n_g = len(idx)

    for scen_name, cov_m, mu_m in [
        ("Baseline",    COV_BASELINE,     MU_BASE),
        ("Choc ESG-aj", COV_SHOCK_ESG,   MU_SHOCK),
    ]:
        w_grp = np.zeros(N)
        w_grp[idx] = 1.0 / n_g

        pvar = portfolio_variance(w_grp, cov_m)
        pret = portfolio_return(w_grp, mu_m)
        group_results.append({
            "ESG Level"   : grp,
            "Scénario"    : scen_name,
            "Volatilité (%/j)": round(np.sqrt(pvar)*100, 4),
            "Rendement (%/j)" : round(pret*100, 4),
            "Sharpe ratio"    : round(pret / np.sqrt(pvar), 3),
        })

grp_df = pd.DataFrame(group_results)
print("\n" + "=" * 65)
print("PERFORMANCE DES PORTEFEUILLES GROUPÉS PAR NIVEAU ESG")
print("=" * 65)
print(grp_df.to_string(index=False))

# Şok döneminde volatilite artışı (%)
print("\nAugmentation de volatilité lors du choc (% relatif) :")
for grp in esg_groups:
    v_base  = grp_df[(grp_df["ESG Level"]==grp) & (grp_df["Scénario"]=="Baseline")]["Volatilité (%/j)"].values[0]
    v_shock = grp_df[(grp_df["ESG Level"]==grp) & (grp_df["Scénario"]=="Choc ESG-aj")]["Volatilité (%/j)"].values[0]
    pct_chg = (v_shock - v_base) / v_base * 100
    print(f"  ESG {grp:6s}: {v_base:.4f}% → {v_shock:.4f}%  (Δ = {pct_chg:+.1f}%)")


# =====================================================================
# HÜCRE 10 — Tasarım çıktısı: Yatırım kararı kuralları
# =====================================================================
print("\n" + "=" * 70)
print("RÈGLES DE DÉCISION D'INVESTISSEMENT (issu de l'optimisation)")
print("=" * 70)

# Optimal ESG-ayarlı portföyde ağırlığı en yüksek firmalar
top_firms = weight_df.sort_values("w_shock_esg (%)", ascending=False).head(5)
print("\nTop-5 firmes recommandées (scénario choc ESG-ajusté) :")
print(top_firms[["ticker","sector","esg_level","esg_total",
                  "esg_s","w_baseline (%)","w_shock_esg (%)","Δw (pp)"]].to_string(index=False))

# ESG eşik önerisi
opt_esg_avg = (w_esg * firms["esg_s"].values).sum()
print(f"\nESG_S moyen du portefeuille optimal (choc) : {opt_esg_avg:.2f}")
print(f"  → Recommandation : cibler ESG_S < {opt_esg_avg:.1f} pour minimiser")
print(f"    l'amplification de volatilité lors des chocs (β₂ = +0.00135***)")




# =====================================================================
# HÜCRE 11 — Excel çıktısı
# =====================================================================
OUT_EXCEL = DATA_DIR / "portfolio_optimization_results.xlsx"

with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
    # Sayfa 1: Firma verileri + ağırlıklar
    weight_df.to_excel(writer, sheet_name="Poids_Optimaux", index=False)

    # Sayfa 2: Senaryo karşılaştırması
    scen_summary = pd.DataFrame([
        {
            "Scénario"       : opt_results[k]["name"],
            "Volatilité (%)" : round(opt_results[k]["std"]*100, 4),
            "Rendement (%)"  : round(opt_results[k]["return"]*100, 4),
            "Sharpe"         : round(opt_results[k]["sharpe"], 3),
        }
        for k in ["baseline","shock_uniform","shock_esg"]
    ])
    scen_summary.to_excel(writer, sheet_name="Scenarios", index=False)

    # Sayfa 3: ESG grup performansı
    grp_df.to_excel(writer, sheet_name="Groupes_ESG", index=False)

    # Sayfa 4: Firma-bazlı şok volatiliteleri
    shock_vol_df = firms[["ticker","sector","esg_level",
                           "esg_total","esg_s","esg_s_z",
                           "vol_base","vol_shock","ret_mean","ret_shock"]].copy()
    shock_vol_df.to_excel(writer, sheet_name="Vol_Choc_Firme", index=False)

print(f"\n✓ Sonuçlar kaydedildi: {OUT_EXCEL}")
print("  Sekmeler: Poids_Optimaux | Scenarios | Groupes_ESG | Vol_Choc_Firme")
print("\n✓ Portföy optimizasyonu tamamlandı.")


