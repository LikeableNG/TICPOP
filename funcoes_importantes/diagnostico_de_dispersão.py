# ============================================================
# DIAGNÓSTICO DE DISPERSÃO
# Por que o gráfico polarimétrico está espalhado? Quais imagens são as piores?
# ------------------------------------------------------------
# Usa o fotometria_filtro_*.csv (e, se pedido, relê as imagens) e, para cada imagem,
# calcula o resíduo em relação ao modelo. Depois cruza esse resíduo com:
#   Ordem      -> piora ao longo da sequência (seeing, foco, atmosfera)
#   Fluxo_rel  -> fluxo total vs mediana (nuvens / transparência)
#   Pico       -> pico da estrela (saturação / não-linearidade)
#   Desloc_P   -> quanto a estrela principal andou (guiagem / rastreio)
#   Desloc_S   -> o feixe secundário NÃO está onde a posição rígida diz (abertura descentrada)
# ============================================================
obs_diag = 2                 # qual Obs_N analisar
filtro_diag = 'r'            # qual filtro analisar
checar_centragem_S = True    # relê as imagens para checar o centro do feixe secundário (mais lento)
n_piores = 12                # quantas imagens piores listar

pasta_diag = f"{pasta_reduced_estrela}/Obs_{obs_diag}_{NOME_ESTRELA}"
csv_diag = [c for c in glob.glob(os.path.join(pasta_diag, 'fotometria_filtro_*.csv'))
            if os.path.basename(c).split('_')[-1].replace('.csv', '').lower() == filtro_diag.lower()]
if not csv_diag:
    raise FileNotFoundError(f"Sem fotometria_filtro_{filtro_diag}.csv em {pasta_diag}")

df = pd.read_csv(csv_diag[0])
df = df[df['Status'] == 'VALIDA'].reset_index(drop=True)
df['Ordem'] = np.arange(len(df))   # ordem em que as imagens foram medidas (nomes ordenados)

# Mesma limpeza da célula "Dados Finais 1"
df = df[(df['Fluxo_Prin'] > 100) & (df['Fluxo_Sec'] > 100)].copy()
df['Zi_raw'] = (df['Fluxo_Sec'] - df['Fluxo_Prin']) / (df['Fluxo_Sec'] + df['Fluxo_Prin'])
df = df[np.abs(df['Zi_raw']) < 0.20].reset_index(drop=True)

# Ajuste SEM rejeição de pontos, para enxergar todos os resíduos
psi = df['Angulo'].values.astype(float)
solver = SLSDualBeamPolarimetry('halfwave', compute_k=True)
res = solver.compute(psi, df['Fluxo_Prin'].values.astype(float), df['Fluxo_Sec'].values.astype(float))
zi = np.asarray(res.zi.nominal, dtype=float)
modelo = np.asarray(res.model(psi * u.deg).nominal, dtype=float)
if len(zi) != len(df):
    raise RuntimeError(f"zi tem {len(zi)} valores e há {len(df)} imagens: não dá para casar resíduo e imagem.")

df['Residuo'] = zi - modelo
mad = 1.4826 * np.median(np.abs(df['Residuo'] - np.median(df['Residuo'])))
df['Resid_sigma'] = (df['Residuo'] - np.median(df['Residuo'])) / mad

# Variáveis candidatas a explicar o resíduo
df['Fluxo_rel'] = (df['Fluxo_Prin'] + df['Fluxo_Sec']) / np.median(df['Fluxo_Prin'] + df['Fluxo_Sec'])
df['Desloc_P'] = np.hypot(df['X_Prin'] - df['X_Prin'].median(), df['Y_Prin'] - df['Y_Prin'].median())

if checar_centragem_S:
    desloc_S = []
    for _, lin in df.iterrows():
        dados = np.squeeze(fits.getdata(os.path.join(pasta_diag, lin['Arquivo'])))
        dados = np.nan_to_num(dados, nan=np.nanmedian(dados), posinf=0, neginf=0)
        xs, ys = rastrear_centroide_ancorado(dados, lin['X_Sec'], lin['Y_Sec'], janela=10)
        desloc_S.append(np.hypot(xs - lin['X_Sec'], ys - lin['Y_Sec']))
    df['Desloc_S'] = desloc_S

# ---------- Resumo ----------
print(f"=== Diagnóstico | Obs {obs_diag} | filtro {filtro_diag.upper()} | {len(df)} imagens ===")
print(f"Espalhamento robusto dos resíduos (MAD): {mad:.5f}")
print(f"Imagens com |resíduo| > 3σ: {(np.abs(df['Resid_sigma']) > 3).sum()}")

pico_max = max(df['Pico_Prin'].max(), df['Pico_Sec'].max())
n_topo = int(((df['Pico_Prin'] >= 0.99 * pico_max) | (df['Pico_Sec'] >= 0.99 * pico_max)).sum())
print(f"Pico máximo: {pico_max:.0f} | imagens com pico a menos de 1% do máximo: {n_topo} "
      f"(vários picos iguais no topo = estrela saturada)")

candidatas = ['Ordem', 'Fluxo_rel', 'Pico_Prin', 'Pico_Sec', 'Desloc_P'] + (['Desloc_S'] if checar_centragem_S else [])
corr = {c: df[c].corr(df['Resid_sigma'].abs(), method='spearman') for c in candidatas}
print("\nCorrelação (Spearman) entre |resíduo| e cada variável (|ρ| > 0,3 já é suspeito; com poucos outliers")
print("isolados ela fica fraca, então olhe também a comparação logo abaixo e a lista das piores imagens):")
for nome, rho in sorted(corr.items(), key=lambda kv: -abs(kv[1]) if kv[1] == kv[1] else 0):
    print(f"   {nome:10s} ρ = {rho:+.2f}")

fora = np.abs(df['Resid_sigma']) > 3
if fora.any() and (~fora).any():
    print("\nMediana nas imagens ruins (|resíduo| > 3σ) vs nas demais (diferença grande = pista):")
    for c in candidatas:
        print(f"   {c:10s} ruins = {df.loc[fora, c].median():10.2f} | demais = {df.loc[~fora, c].median():10.2f}")

cols = ['Arquivo', 'Angulo', 'Resid_sigma', 'Fluxo_rel', 'Pico_Prin', 'Desloc_P'] + (['Desloc_S'] if checar_centragem_S else [])
piores = df.reindex(df['Resid_sigma'].abs().sort_values(ascending=False).index).head(n_piores)
print(f"\nAs {n_piores} piores imagens:")
print(piores[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# ---------- Gráficos ----------
graficos = [(c, c) for c in candidatas if c != 'Pico_Sec']
fig, eixos = plt.subplots(2, 3, figsize=(14, 7))
for ax, (col, rotulo) in zip(eixos.ravel(), graficos):
    ax.scatter(df.loc[~fora, col], df.loc[~fora, 'Resid_sigma'], s=14, color='navy', alpha=0.7)
    ax.scatter(df.loc[fora, col], df.loc[fora, 'Resid_sigma'], s=22, color='crimson', label='|resíduo| > 3σ')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel(rotulo); ax.set_ylabel('resíduo (σ)')
    ax.grid(True, linestyle=':', alpha=0.6)
eixos.ravel()[0].legend(loc='best', fontsize=8)
for ax in eixos.ravel()[len(graficos):]:
    ax.axis('off')
fig.suptitle(f"Diagnóstico de dispersão - filtro {filtro_diag.upper()} (Obs {obs_diag})")
plt.tight_layout()
plt.savefig(os.path.join(pasta_diag, f'diagnostico_dispersao_filtro_{filtro_diag}.png'), dpi=200, bbox_inches='tight')
plt.show()
