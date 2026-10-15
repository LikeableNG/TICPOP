import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropop.polarimetry import SLSDualBeamPolarimetry

pasta_obj_reduced = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/paqr/'
arquivos_csv = sorted(glob.glob(os.path.join(pasta_obj_reduced, 'fotometria_filtro_*.csv')))

if not arquivos_csv:
    raise FileNotFoundError("Nenhum arquivo 'fotometria_filtro_*.csv' foi encontrado.")

# Inicializa o ajustador dos Mínimos Quadrados de Stokes para lâmina de Meia-Onda (Half-Wave)
# compute_k=True calcula o fator de normalização entre os feixes automaticamente
pol_solver = SLSDualBeamPolarimetry('halfwave', compute_k=True)

for csv_file in arquivos_csv:
    filtro = os.path.basename(csv_file).split('_')[-1].replace('.csv', '').upper()
    df = pd.read_csv(csv_file)
    
    # Extrai as matrizes de ângulos e fluxos
    psi = df['Angulo'].values
    fo = df['Fluxo_Prin'].values
    fe = df['Fluxo_Sec'].values
    
    # Executa o ajuste dos Mínimos Quadrados no astropop
    resultado = pol_solver.compute(psi, fo, fe)
    
    print(f"\n==========================================")
    print(f"  RESULTADOS DE POLARIMETRIA - FILTRO {filtro}")
    print(f"==========================================")
    print(f"Parâmetro q  : {resultado.q}")
    print(f"Parâmetro u  : {resultado.u}")
    print(f"Grau de Pol. (P) : {resultado.p * 100}%")  # Converte fração para %
    print(f"Ângulo (Theta)   : {resultado.theta}")
    print(f"Fator Norm. (k)  : {resultado.k:.4f}")
    print(f"Resíduo (RMS)    : {resultado.rms:.6f}")
    
    # --- GRÁFICO DA CURVA DE MODULAÇÃO Z_i ---
    plt.figure(figsize=(8, 5))
    
    # Plota os dados medidos Z_i
    plt.errorbar(resultado.psi.nominal, resultado.zi.nominal, yerr=resultado.zi.std_dev, 
                 fmt='o', color='crimson', label='Dados Medidos ($Z_i$)')
    
    # Plota o modelo ajustado pelo astropop
    eixo_psi = np.linspace(0, 360, 300)
    modelo_zi = resultado.model(eixo_psi * u.deg).nominal
    plt.plot(eixo_psi, modelo_zi, '--', color='navy', label='Modelo SLS ($Q\cos 4\psi + U\sin 4\psi$)')
    
    plt.title(f"Curva de Modulação Polarimétrica - Filtro {filtro}\n$P = {resultado.p.nominal*100:.2f}\% \quad \\theta = {resultado.theta.nominal:.1f}^\circ$")
    plt.xlabel(r"Posição da Lâmina Retardadora $\psi$ (graus)")
    plt.ylabel(r"Diferença Relativa de Fluxo $Z_i = \frac{F_e - F_o \cdot k}{F_e + F_o \cdot k}$")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()