# # 5. Metodoloji: İki-Yönlü Sabit Etkiler Panel Regresyonu
# 
# ## 5.1 Model Spesifikasyonu
# 
# **Two-Way Fixed Effects (TWFE)** modeli, firma ve zaman boyutlarındaki gözlemlenemeyen heterojenliği aynı anda kontrol eder:
# 
# $$Y_{it} = \alpha + \underbrace{\beta_1 \cdot \text{Shock}_{it}}_{\text{doğrudan etki}} + \underbrace{\beta_2 \cdot (\text{Shock}_{it} \times \text{ESG}_i)}_{\text{moderatör etki}} + \beta_3 \cdot \log(\text{news}_{it}) + \mu_i + \lambda_t + \varepsilon_{it}$$
# 
# | Bileşen | Açıklaması |
# |---|---|
# | $\mu_i$ | Firma sabit etkisi (firma-spesifik sabit faktörleri absorbe etmesi için) |
# | $\lambda_t$ | Zaman sabit etkisi ( tüm firmayı etkileyen makroekonomik faktörler) |
# | $\varepsilon_{it}$ | Cluster-robust standart hata (firma düzeyinde) |
# 
# ## 5.2 Uygulanan 8 Ana Model
# 
# | Model | Bağımlı değişken (Y) | ESG Bileşeni |
# |---|---|---|
# | M1 | return_1d | ESG Total |
# | M2 | return_1d | ESG Environmental (E) |
# | M3 | return_1d | ESG Social (S) |
# | M4 | return_1d | ESG Governance (G) |
# | M5 | volatility | ESG Total |
# | M6 | volatility | ESG Environmental (E) |
# | M7 | volatility | ESG Social (S) |
# | M8 | volatility | ESG Governance (G) |
# 
# ## 5.3 ESG Standardizasyonu
# 
# E, S, G bileşenleri farklı ölçeklerde olduğundan (E: 0–18, S: 5–21, G: 3–19, Total: 13–36), tüm ESG değişkenleri firma düzeyinde standardize edilmiştir:
# 
# $$\text{ESG}_i^z = \frac{\text{ESG}_i - \bar{\text{ESG}}}{\text{SD}(\text{ESG})}$$
# 
# Bu sayede $\beta_2$ katsayısı *"1 standart sapmalık ESG artışı başına moderatör etki"* olarak yorumlanabilmekle birlikte dört ayrı bileşen karşılaştırılabilir hale gelmektedir.

from linearmodels.panel import PanelOLS

# Regresyon için panelin uüklenmesi
xl        = pd.ExcelFile(OUT_PANEL)
panel_reg = pd.read_excel(OUT_PANEL, sheet_name=xl.sheet_names[0])
panel_reg["date"] = pd.to_datetime(panel_reg["date"])

if "n_news" in panel_reg.columns and "news_count" not in panel_reg.columns:
    panel_reg = panel_reg.rename(columns={"n_news": "news_count"})

# Esg risk skorlarının stanzartize edilmesi
esg_components = ["esg_total", "esg_e", "esg_s", "esg_g"]
esg_firm = panel_reg.drop_duplicates("ticker")[["ticker"] + esg_components].copy()

for c in esg_components:
    mu, sd = esg_firm[c].mean(), esg_firm[c].std()
    esg_firm[f"{c}_z"] = (esg_firm[c] - mu) / sd

print("Standardize ESG (12 firma):")
print(esg_firm[["ticker"] + [f"{c}_z" for c in esg_components]].round(2).to_string(index=False))

panel_reg = panel_reg.merge(
    esg_firm[["ticker"] + [f"{c}_z" for c in esg_components]],
    on="ticker", how="left"
)

# İnteraksiyon: Shock × ESG_z
for comp in ["total", "e", "s", "g"]:
    panel_reg[f"shock_x_esg_{comp}"] = panel_reg["shock"] * panel_reg[f"esg_{comp}_z"]

# Logaritmik dönüşüm uygulanmış haber sayısı (right-skewed distributin)
panel_reg["log_news"] = np.log1p(panel_reg["news_count"])

# Lineer model için Çoklu İndeks Ayarı PanelOLS
panel_reg = panel_reg.set_index(["ticker", "date"]).sort_index()

print(f"\n✓ Panel regression için hazır: {panel_reg.shape}")

def run_twoway_fe(data, y_col, esg_component):
    """
    Estimate Two-Way Fixed Effects panel regression.

    Parameters
    ----------
    data          : MultiIndex (ticker, date) DataFrame
    y_col         : dependent variable column name
    esg_component : 'total' | 'e' | 's' | 'g'

    Returns
    -------
    PanelEffectsResults with cluster-robust SE at firm level
    """
    interaction = f"shock_x_esg_{esg_component}"
    cols = [y_col, "shock", interaction, "log_news"]
    df   = data[cols].dropna()

    model = PanelOLS(
        dependent      = df[y_col],
        exog           = df[["shock", interaction, "log_news"]],
        entity_effects = True,
        time_effects   = True,
        drop_absorbed  = True,
    )
    return model.fit(cov_type="clustered", cluster_entity=True)


def result_row(res, label, esg_component):
    """Extract a summary dict from regression results."""
    interaction = f"shock_x_esg_{esg_component}"

    def fmt(coef, pval):
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        return f"{coef:.5f}{stars}"

    return {
        "Model"        : label,
        "N"            : int(res.nobs),
        "R²_within"    : round(res.rsquared_within, 4),
        "β1_shock"     : fmt(res.params["shock"],       res.pvalues["shock"]),
        "β1_SE"        : f"({res.std_errors['shock']:.5f})",
        "β2_ShockxESG" : fmt(res.params[interaction],   res.pvalues[interaction]),
        "β2_SE"        : f"({res.std_errors[interaction]:.5f})",
        "β2_pvalue"    : round(res.pvalues[interaction], 4),
        "β2_sig"       : ("***" if res.pvalues[interaction] < 0.01 else
                          "**"  if res.pvalues[interaction] < 0.05 else
                          "*"   if res.pvalues[interaction] < 0.10 else ""),
    }


print("=" * 70)
print("PILOT: return_1d ~ shock + shock×ESG_total + log_news + FE")
print("=" * 70)
pilot = run_twoway_fe(panel_reg, "return_1d", "total")
print(pilot.summary)

