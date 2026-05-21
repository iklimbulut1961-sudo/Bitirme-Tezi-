# # 7. Görselleştirmeler / Visualisations
# 
# Bu bölümde 6 akademik figür üretilmektedir.
# 
# | Figür | İçerik |
# |---|---|
# | Fig. 1 | Aylık şok dağılımı + sektörel pasta grafiği |
# | Fig. 2 | FinBERT polarite zaman serisi (4 temsili firma) |
# | Fig. 3 | ESG skor dağılımı sektöre göre |
# | Fig. 4 | ESG seviyesine göre şok günü getirileri |
# | Fig. 5 | Firma × ay volatilite ısı haritası |
# | Fig. 6 | β₂ katsayı grafiği (8 model) |


NAVY  = '#1F3864'
TEAL  = '#1F6B75'
GOLD  = '#C9A84C'
RED   = '#C0392B'
GRAY  = '#95A5A6'

SECTOR_COLORS = {
    'Energy':             NAVY,
    'Financial Services': TEAL,
    'Healthcare':         GOLD,
    'Technology':         '#8E44AD'
}
ESG_COLORS = {'High': '#C0392B', 'Medium': '#C9A84C', 'Low': '#27AE60'}

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
})

df = pd.read_excel(F_PANEL_VIZ, header=4)
df.columns = [
    'Ticker', 'Date', 'Sector', 'ESG_Level',
    'News_Count', 'Positive', 'Negative', 'Neutral',
    'Avg_Polarity', 'Avg_Confidence', 'Neg_Ratio',
    'Rolling_ZScore', 'Shock_Intensity', 'Shock',
    'Price', 'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d',
    'Volatility', 'ESG_Total', 'ESG_E', 'ESG_S', 'ESG_G',
    'Controversy_Level', 'Controversy_Score', 'ESG_Percentile'
]
df = df[df['Ticker'].notna() & (df['Ticker'] != 'Ticker') & (df['Ticker'].str.len() <= 5)]
numeric_cols = ['Return_1d','Return_3d','Return_5d','Return_10d','Volatility',
                'ESG_Total','ESG_E','ESG_S','ESG_G','News_Count','Shock','Avg_Polarity','Rolling_ZScore']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.to_period('M')
print(f"Görselleştirme verisi: {df.shape[0]:,} gözlem, {df['Ticker'].nunique()} firma")
print(f"Toplam şok: {int(df['Shock'].sum())} (%{df['Shock'].mean()*100:.2f})")

# Figure 1 — Aylık şok dağılımı + sektörel pie chart
print("Figure 1: Şok dağılımı...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Solda: Aylık bar chart
monthly = df.groupby('Month')['Shock'].sum().reset_index()
monthly['Month_dt'] = monthly['Month'].dt.to_timestamp()
ax = axes[0]
bars = ax.bar(monthly['Month_dt'], monthly['Shock'], width=20,
              color=NAVY, alpha=0.85, edgecolor='white', linewidth=0.5)
ax.axvspan(pd.Timestamp('2023-03-01'), pd.Timestamp('2023-04-01'),
           alpha=0.12, color=RED, label='Crise SVB (mars 2023)')
for bar in bars:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15, str(int(h)),
                ha='center', va='bottom', fontsize=7.5, color=NAVY, fontweight='bold')
ax.set_xlabel('Mois (2023)', labelpad=6)
ax.set_ylabel('Nombre de chocs de sentiment négatif', labelpad=6)
ax.set_title('Distribution mensuelle des chocs de sentiment négatif', fontweight='bold', pad=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_xlim(pd.Timestamp('2022-12-20'), pd.Timestamp('2023-12-31'))
ax.legend(fontsize=8, framealpha=0.7)

# Sağda: Sektörel pie chart
sector_shocks = df.groupby('Sector')['Shock'].sum()
vals   = [sector_shocks.get(s, 0) for s in ['Energy','Financial Services','Healthcare','Technology']]
labels = [f"Énergie\n({int(vals[0])})", f"Finance\n({int(vals[1])})",
          f"Santé\n({int(vals[2])})",    f"Technologie\n({int(vals[3])})"]
wedges, texts, autotexts = axes[1].pie(
    vals, labels=labels, colors=[NAVY, TEAL, GOLD, '#8E44AD'],
    explode=[0.03, 0.06, 0.03, 0.03], autopct='%1.1f%%', pctdistance=0.75,
    startangle=140, textprops={'fontsize': 9}
)
for at in autotexts:
    at.set_fontsize(8); at.set_color('white'); at.set_fontweight('bold')
axes[1].set_title('Répartition des chocs par secteur (N = 131)', fontweight='bold', pad=10)

plt.tight_layout(pad=2)
plt.savefig('figures/fig1_shock_distribution.pdf', bbox_inches='tight')
plt.show()
print("fig1_shock_distribution.pdf")

# **Figure 1**  
# Sol panel, 2023 yılı boyunca tespit edilen 131 negatif sentiment şokunun aylık dağılımını göstermektedir. Mart 2023'te gözlemlenen yoğunlaşma, Silicon Valley Bank iflasıyla eş zamanlıdır. Sağ panel, şokların sektörler arasındaki dağılımını verir; Financial Services ve Technology sektörleri öne çıkmıştır.

# Figure 2 — FinBERT pPolarite zaman serisi (4 firma)
print("Figure 2: Polarite zaman serisi...")
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
sample_firms = [('NVDA','Technologie'), ('WFC','Finance'), ('LLY','Santé'), ('COP','Énergie')]

for ax, (ticker, sector_label) in zip(axes.flatten(), sample_firms):
    sub    = df[df['Ticker'] == ticker].sort_values('Date')
    shocks = sub[sub['Shock'] == 1]
    rolling = sub.set_index('Date')['Avg_Polarity'].rolling('14D').mean()

    ax.fill_between(sub['Date'], sub['Avg_Polarity'], 0,
                    where=(sub['Avg_Polarity'] >= 0), alpha=0.20, color=TEAL, interpolate=True)
    ax.fill_between(sub['Date'], sub['Avg_Polarity'], 0,
                    where=(sub['Avg_Polarity'] < 0), alpha=0.20, color=RED, interpolate=True)
    ax.plot(rolling.index, rolling.values, color=NAVY, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.6)

    for _, row in shocks.iterrows():
        ax.axvline(row['Date'], color=RED, alpha=0.5, linewidth=0.7, linestyle=':')

    ax.set_title(f'{ticker} — {sector_label}', fontweight='bold', fontsize=10, pad=6)
    ax.set_ylabel('Polarité', fontsize=9)

legend_elems = [
    Line2D([0],[0], color=NAVY, linewidth=1.5, label='MM14 (polarité)'),
    Patch(color=TEAL, alpha=0.4, label='Zone positive'),
    Patch(color=RED,  alpha=0.4, label='Zone négative'),
    Line2D([0],[0], color=RED, linewidth=1, linestyle=':', label='Choc détecté'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Évolution de la polarité de sentiment FinBERT (2023)\n'
             'Moyenne mobile 14j | Chocs indiqués par trait pointillé rouge',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.savefig('figures/fig2_polarity_timeseries.pdf', bbox_inches='tight')
plt.show()
print("fig2_polarity_timeseries.pdf")

# Figure 3 — Sektör bazlı ESG skor dağılımı
print("Figure 3: ESG skor dağılımı...")
firm_esg = df.drop_duplicates('Ticker')[['Ticker','Sector','ESG_Total','ESG_E','ESG_S','ESG_G','ESG_Level']].copy()
firm_esg = firm_esg.sort_values('ESG_Total')

esg_components = {'ESG_E': 'Environmental (E)', 'ESG_S': 'Social (S)', 'ESG_G': 'Governance (G)'}
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, (col, title) in zip(axes, esg_components.items()):
    colors = [ESG_COLORS.get(lev, GRAY) for lev in firm_esg['ESG_Level']]
    bars = ax.bar(firm_esg['Ticker'], firm_esg[col], color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    ax.set_title(f'Score de risque {title}', fontweight='bold', pad=8)
    ax.set_ylabel('Score de risque ESG')
    ax.set_xlabel('Entreprise')
    ax.tick_params(axis='x', rotation=45)

legend_elems = [Patch(color=c, label=f'ESG {k}') for k, c in ESG_COLORS.items()]
fig.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)
fig.suptitle('Distribution des scores de risque ESG par composante (12 entreprises, 2023)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig3_esg_scores.pdf', bbox_inches='tight')
plt.show()
print("fig3_esg_scores.pdf")

# Figure 4 — ESG level × shock day returns
print("Figure 4: ESG × getiri karşılaştırması...")
horizons       = ['Return_1d','Return_3d','Return_5d','Return_10d']
horizon_labels = ['1j','3j','5j','10j']
esg_groups     = ['High','Medium','Low']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Solda: 1-günlük getiri shock vs normal
ax = axes[0]
shock_means   = [df[(df['ESG_Level']==g)&(df['Shock']==1)]['Return_1d'].mean()*100 for g in esg_groups]
noshock_means = [df[(df['ESG_Level']==g)&(df['Shock']==0)]['Return_1d'].mean()*100 for g in esg_groups]
x = np.arange(3); w = 0.35
b1 = ax.bar(x-w/2, noshock_means, w, label='Jours normaux', color=TEAL, alpha=0.85)
b2 = ax.bar(x+w/2, shock_means,   w, label='Jours de choc', color=RED,  alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['ESG High\n(risque élevé)','ESG Medium','ESG Low\n(risque faible)'])
ax.set_ylabel('Rendement moyen à 1 jour (%)')
ax.set_title('Rendement moyen à 1 jour selon le niveau ESG\n(jours normaux vs jours de choc)', fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0, color='black', linewidth=0.6)
for bar in list(b1)+list(b2):
    h = bar.get_height(); off = 0.008 if h>=0 else -0.016; va = 'bottom' if h>=0 else 'top'
    ax.text(bar.get_x()+bar.get_width()/2, h+off, f'{h:.3f}%',
            ha='center', va=va, fontsize=7.5, fontweight='bold')

# Sağda: Şok günlerinde kümülatif return trajektoire'ı
ax = axes[1]
for esg_lev in esg_groups:
    sub   = df[(df['ESG_Level']==esg_lev)&(df['Shock']==1)]
    means = [sub[h].mean()*100 for h in horizons]
    ax.plot(horizon_labels, means, marker='o', color=ESG_COLORS[esg_lev],
            linewidth=2, markersize=8, label=f'ESG {esg_lev}', zorder=3)
    for xi, yi in enumerate(means):
        ax.annotate(f'{yi:.3f}%', (xi, yi), textcoords='offset points',
                    xytext=(0, 9 if yi>=0 else -14), ha='center', fontsize=7.5, color=ESG_COLORS[esg_lev])
ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
ax.set_ylabel('Rendement cumulé moyen (%)')
ax.set_title('Trajectoire des rendements cumulés lors des chocs\nselon le niveau de risque ESG', fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlabel('Horizon temporel')

plt.tight_layout(pad=2)
plt.savefig('figures/fig4_returns_esg.pdf', bbox_inches='tight')
plt.show()
print("fig4_returns_esg.pdf")

# Figure 5 — Volatility heatmap (firm × month)
print("Figure 5: Volatilite heatmap...")
pivot = df.pivot_table(values='Volatility', index='Ticker',
                        columns=df['Date'].dt.month, aggfunc='mean')
pivot.columns = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
firm_order = ['NVDA','VRSN','MCHP', 'WFC','GS','V', 'LLY','UNH','UHS', 'COP','KMI','OKE']
pivot = pivot.reindex([t for t in firm_order if t in pivot.index])

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.3f', annot_kws={'size': 8},
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Volatilité réalisée (écart-type 20j)'})
ax.set_title('Carte thermique de la volatilité réalisée par entreprise et par mois (2023)\n'
             'Groupes : Technologie | Finance | Santé | Énergie',
             fontweight='bold', pad=12)
ax.set_xlabel('Mois', labelpad=6)
ax.set_ylabel('Entreprise', labelpad=6)
for y in [3, 6, 9]:
    ax.axhline(y, color=NAVY, linewidth=1.5)
plt.tight_layout()
plt.savefig('figures/fig5_volatility_heatmap.pdf', bbox_inches='tight')
plt.show()
print("fig5_volatility_heatmap.pdf")

# Figure 6 — Katsayı Grafiği (8 model için β₂)
print("Figure 6: Katsayı grafiği...")

# Two-Way FE regresyonundan gelen sonuçlar (cluster-robust SE)
models_ret = ['ESG Total\n(M1)','ESG E\n(M2)','ESG S\n(M3)','ESG G\n(M4)']
beta2_ret  = [ 0.00217,  0.00119,  0.00114,  0.00132]
se_ret     = [ 0.00079,  0.00070,  0.00096,  0.00085]
pval_ret   = [ 0.006,    0.087,    0.236,    0.122  ]

models_vol = ['ESG Total\n(M5)','ESG E\n(M6)','ESG S\n(M7)','ESG G\n(M8)']
beta2_vol  = [ 0.00032, -0.00064,  0.00135,  0.00033]
se_vol     = [ 0.00064,  0.00043,  0.00043,  0.00032]
pval_vol   = [ 0.618,    0.132,    0.002,    0.305  ]

def bar_color(p):
    return NAVY if p < 0.05 else GOLD if p < 0.10 else GRAY

def sig_star(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
specs = [
    (axes[0], models_ret, beta2_ret, se_ret, pval_ret,
     'Effet modérateur ESG sur\nle rendement à 1 jour (β₂)'),
    (axes[1], models_vol, beta2_vol, se_vol, pval_vol,
     'Effet modérateur ESG sur\nla volatilité (β₂)'),
]

for ax, models, betas, ses, pvals, title in specs:
    y = np.arange(len(models))
    ci95 = [1.96*s for s in ses]
    colors = [bar_color(p) for p in pvals]
    ax.barh(y, betas, color=colors, alpha=0.85, height=0.5, xerr=ci95,
            error_kw={'ecolor':'#2C3E50','capsize':5,'linewidth':1.3,'capthick':1.3})
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_yticks(y); ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel('Coefficient β₂ (IC 95%)')
    ax.set_title(title, fontweight='bold', pad=10)
    for i, (b, s, p) in enumerate(zip(betas, ses, pvals)):
        p_label = f'p={p:.3f}' if p >= 0.001 else 'p<0,001'
        ax.text(b/2, i, p_label, va='center', ha='center', fontsize=7,
                color='white' if abs(b) > 0.0004 else 'black')
        star = sig_star(p)
        if star:
            ax.text(b + 1.96*s + abs(max(betas))*0.06, i, star,
                    va='center', fontsize=11, color=RED, fontweight='bold')

legend_elems = [
    Patch(color=NAVY, label='Significatif (p < 0,05)'),
    Patch(color=GOLD, label='Marginalement sig. (p < 0,10)'),
    Patch(color=GRAY, label='Non significatif (p ≥ 0,10)'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=3,
           framealpha=0.9, fontsize=9, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Graphique des coefficients β₂ — Effets modérateurs de l'ESG (TWFE)\n"
             "Barres d'erreur : intervalle de confiance à 95% (SE cluster-robustes)",
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0.09, 1, 0.95])
plt.savefig('figures/fig6_coefficient_plot.pdf', bbox_inches='tight')
plt.show()
print("fig6_coefficient_plot.pdf")

# **Figure 6**  
# Sol panel 1 günlük getiri modelleri için (M1–M4), sağ panel ise volatilite modelleri için (M5–M8) $\beta_2$ moderatör katsayılarını 95% güven aralıkları ile göstermektedir (a= %5). Koyu mavi çubuklar istatistiksel olarak anlamlı moderatör etkiyi (p<0.05) temsil etmektedir. ESG_S (M7) volatilite üzerinde anlamlı pozitif bir moderatör etki sergilemektedir yani yüksek sosyal risk skoru olan firmalar şok günlerinde daha yüksek volatilite yaşamaktadır.

