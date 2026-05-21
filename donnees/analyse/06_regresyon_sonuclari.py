# # 6. Regresyon Sonuçları
# 
# ## 6.1 Ana Modeller (8 Regresyon)
# 
# 8 ana modeli paralel olarak çalıştırdık ve sonuçları derledik. Her model için raporlanan $\beta_2$ katsayısı, ESG risk skorunun moderatör etkisini temsil etmektedir:
# 
# - $\beta_2 > 0$: Yüksek ESG riski, şokun olumsuz etkisini **ağırlaştırır** (hipotezimizle tutarsız)
# - $\beta_2 < 0$: Yüksek ESG riski, şokun olumsuz etkisini **hafifletir** — *istenilen çıktı*
# 
# **Önem seviyeleri:** `***` p<0.01 | `**` p<0.05 | `*` p<0.10

specs_main = [
    ("return_1d",  "total", "M1: return_1d  × ESG_total"),
    ("return_1d",  "e",     "M2: return_1d  × ESG_E"),
    ("return_1d",  "s",     "M3: return_1d  × ESG_S"),
    ("return_1d",  "g",     "M4: return_1d  × ESG_G"),
    ("volatility", "total", "M5: volatility × ESG_total"),
    ("volatility", "e",     "M6: volatility × ESG_E"),
    ("volatility", "s",     "M7: volatility × ESG_S"),
    ("volatility", "g",     "M8: volatility × ESG_G"),
]

fitted_main = {}
rows_main   = []
for y_col, comp, label in specs_main:
    r = run_twoway_fe(panel_reg, y_col, comp)
    fitted_main[label] = r
    rows_main.append(result_row(r, label, comp))

df_main = pd.DataFrame(rows_main)

print("\n" + "=" * 110)
print("ANA MODELLER — 8 Regresyon (Two-Way FE, Cluster-Robust SE, firma düzeyinde)")
print("=" * 110)
print(df_main.to_string(index=False))
print("\nAnlamlılık: *** p<0.01  ** p<0.05  * p<0.10")
print("β2 > 0 → yüksek ESG riski şok etkisini AĞIRLAŞTIRIR")
print("β2 < 0 → yüksek ESG riski şok etkisini AZALTIR (beklenen yön)")

# ## 6.2 Robustness Analizi: 3, 5 ve 10 Günlük Ufuklar
# 
# Sentiment şoklarının etkisi anlık olmayabilir; piyasa tepkisi birkaç günde tamamlanıyor olabilir. Bu nedenle 1 günlük getiri sonuçlarımızı 3, 5 ve 10 günlük kümülatif getiri ufuklarında **tekrarlıyoruz** (12 ek model).
# 
# Sonuçların zaman ufuklarında tutarlılığı, bulgularımızın robustluğunun göstergesi olacaktır.

# Robustness: return_3d / return_5d / return_10d
specs_robust = [
    (y, comp, f"{y} × ESG_{comp}")
    for y    in ["return_3d", "return_5d", "return_10d"]
    for comp in ["total", "e", "s", "g"]
]

rows_robust = []
for y_col, comp, label in specs_robust:
    r = run_twoway_fe(panel_reg, y_col, comp)
    rows_robust.append(result_row(r, label, comp))

df_robust = pd.DataFrame(rows_robust)
print("\n" + "=" * 110)
print("ROBUSTNESS — return_3d / 5d / 10d (12 Model)")
print("=" * 110)
print(df_robust.to_string(index=False))


print("\n" + "=" * 75)
print("HİPOTEZ TESTİ SONUÇLARI")
print("=" * 75)

hypotheses = {
    "M1: return_1d  × ESG_total" : ("H2  — ESG_total moderasyon",  "total"),
    "M2: return_1d  × ESG_E"     : ("H2a — Environment moderasyon", "e"),
    "M3: return_1d  × ESG_S"     : ("H2b — Social moderasyon",      "s"),
    "M4: return_1d  × ESG_G"     : ("H2c — Governance moderasyon",  "g"),
}

print(f"\n{'Hipotez':<35} {'β2':>10} {'SE':>10} {'p-value':>10} Karar")
print("-" * 80)
for label, (hyp, comp) in hypotheses.items():
    r   = fitted_main[label]
    ix  = f"shock_x_esg_{comp}"
    b2  = r.params[ix]
    se  = r.std_errors[ix]
    p   = r.pvalues[ix]
    dec = "DESTEKLENDİ ✓" if p < 0.10 else "reddedildi"
    print(f"{hyp:<35} {b2:>10.5f} {se:>10.5f} {p:>10.4f}   {dec}")

print(f"\n{'H1 — Shock doğrudan etkisi':<35} {'β1':>10} {'SE':>10} {'p-value':>10}")
print("-" * 65)
for label in ["M1: return_1d  × ESG_total", "M5: volatility × ESG_total"]:
    r  = fitted_main[label]
    b1 = r.params["shock"]
    se = r.std_errors["shock"]
    p  = r.pvalues["shock"]
    print(f"  {label:<33} {b1:>10.5f} {se:>10.5f} {p:>10.4f}")

# Excel Kaydı
with pd.ExcelWriter(OUT_RESULTS, engine="openpyxl") as w:
    df_main.to_excel(w, sheet_name="Ana_Modeller_8", index=False)
    df_robust.to_excel(w, sheet_name="Robustness_12", index=False)

    all_rows = []
    for label, r in fitted_main.items():
        for var in r.params.index:
            all_rows.append({
                "Model": label, "Variable": var,
                "Coef": r.params[var], "SE": r.std_errors[var],
                "t": r.tstats[var], "p": r.pvalues[var],
            })
    pd.DataFrame(all_rows).to_excel(w, sheet_name="Tüm_Katsayılar", index=False)

print(f" Kaydedildi: {OUT_RESULTS}")
print("  Sekmeler: Ana_Modeller_8 | Robustness_12 | Tüm_Katsayılar")

