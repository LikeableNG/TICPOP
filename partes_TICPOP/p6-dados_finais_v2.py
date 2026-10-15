import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.stats import sigma_clip
from astropop.polarimetry import SLSDualBeamPolarimetry

pasta_reduced_estrela = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/paqr/'
arquivos_csv = sorted(glob.glob(os.path.join(pasta_reduced_estrela, 'fotometria_filtro_*.csv')))

if not arquivos_csv:
    raise FileNotFoundError("Nenhum arquivo CSV de fotometria foi encontrado.")

# Solver de Mínimos Quadrados do astropop para lâmina de meia-onda
pol_solver = SLSDualBeamPolarimetry('halfwave', compute_k=True)

resultados_finais = []

for csv_file in arquivos_csv:
    filtro = os.path.basename(csv_file).split('_')[-1].replace('.csv', '').upper()
    df = pd.read_csv(csv_file)
    
    # 1. Filtra quadros com contaminação severa/fluxos inválidos
    df_valido = df[(df['Fluxo_Prin'] > 100) & (df['Fluxo_Sec'] > 100)].copy()
    
    # 2. Descarte primário de dispersões grotescas (|Zi| >= 0.20)
    df_valido['Zi_raw'] = (df_valido['Fluxo_Sec'] - df_valido['Fluxo_Prin']) / (df_valido['Fluxo_Sec'] + df_valido['Fluxo_Prin'])
    df_limpo = df_valido[np.abs(df_valido['Zi_raw']) < 0.20].copy()
    
    if len(df_limpo) == 0:
        print(f"[{filtro}] Nenhum dado válido restou após filtros primários.")
        continue

    psi = df_limpo['Angulo'].values
    fo = df_limpo['Fluxo_Prin'].values
    fe = df_limpo['Fluxo_Sec'].values
    
    # 3. Ajuste preliminar no astropop
    res_preliminar = pol_solver.compute(psi, fo, fe)
    
    # 4. Rejeição refinada por 3-Sigma nos resíduos do modelo
    residuos = res_preliminar.zi.nominal - res_preliminar.model(psi * u.deg).nominal
    mascara_boa = ~sigma_clip(residuos, sigma=3.0).mask
    
    # 5. Ajuste Final de Mínimos Quadrados
    resultado_final = pol_solver.compute(psi[mascara_boa], fo[mascara_boa], fe[mascara_boa])
    
    n_total = len(df)
    n_usados = np.sum(mascara_boa)
    
    p_pct = resultado_final.p.nominal * 100
    p_err = resultado_final.p.std_dev * 100
    theta_deg = resultado_final.theta.nominal
    theta_err = resultado_final.theta.std_dev
    
    resultados_finais.append({
        'Filtro': filtro,
        'N_Usados': f"{n_usados}/{n_total}",
        'P (%)': f"{p_pct:.3f} ± {p_err:.3f}",
        'Theta (°)': f"{theta_deg:.2f} ± {theta_err:.2f}",
        'k': f"{resultado_final.k:.4f}",
        'RMS': f"{resultado_final.rms:.6f}"
    })
    
    # 6. Plotagem da Curva de Modulação
    plt.figure(figsize=(8, 5))
    
    plt.errorbar(resultado_final.psi.nominal, resultado_final.zi.nominal, 
                 yerr=resultado_final.zi.std_dev, fmt='o', color='crimson', 
                 alpha=0.7, label=f'Medições Válidas (N={n_usados})')
    
    eixo_psi = np.linspace(0, 360, 300)
    modelo_zi = resultado_final.model(eixo_psi * u.deg).nominal
    plt.plot(eixo_psi, modelo_zi, '--', color='navy', lw=2, label='Modelo SLS ($Q\cos 4\psi + U\sin 4\psi$)')
    
    plt.title(f"Curva Polarimétrica Limpa - Filtro {filtro}\n$P = {p_pct:.2f}\% \pm {p_err:.2f}\% \quad \\theta = {theta_deg:.1f}^\circ \pm {theta_err:.1f}^\circ$")
    plt.xlabel(r"Posição da Lâmina Retardadora $\psi$ (graus)")
    plt.ylabel(r"Diferença Relativa de Fluxo $Z_i$")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Resumo final em tabela no terminal
print("\n" + "="*70)
print("  RESUMO DA POLARIMETRIA FINAL (TODOS OS FILTROS)")
print("="*70)
df_resumo = pd.DataFrame(resultados_finais)
print(df_resumo.to_string(index=False))