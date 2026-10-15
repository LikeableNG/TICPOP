import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, aperture_photometry

# ==========================================
# 1. CONFIGURAÇÃO E AGRUPAMENTO DE ARQUIVOS
# ==========================================
pasta_obj_reduced = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/osco/'

# Busca TODOS os arquivos reduzidos que começam com 'red_'
todos_arquivos = glob.glob(os.path.join(pasta_obj_reduced, 'red_*.fits'))

if not todos_arquivos:
    raise ValueError("Nenhum arquivo 'red_*.fits' encontrado na pasta de objetos reduzidos.")

# Agrupa os arquivos por filtro em um dicionário
arquivos_por_filtro = defaultdict(list)
for arq in todos_arquivos:
    nome = os.path.basename(arq)
    # Exemplo: red_osco_i_015_030.fits -> partes = ['red', 'osco', 'i', '015', '030.fits']
    partes = nome.split('_')
    if len(partes) >= 5:
        filtro = partes[2]  # O filtro é o 3º elemento (índice 2)
        arquivos_por_filtro[filtro].append(arq)

print("=== PIPELINE DE FOTOMETRIA BEACON ===")
print(f"Filtros identificados: {list(arquivos_por_filtro.keys())}\n")

raio_abertura = 6.0    
tamanho_caixa_rastreio = 15 
meio = tamanho_caixa_rastreio // 2

def rastrear_centroide(dados, x_atual, y_atual):
    """Calcula o centroide subpixel usando uma micro-janela de rastreamento."""
    x_int, y_int = int(round(x_atual)), int(round(y_atual))
    recorte = dados[y_int-meio : y_int+meio+1, x_int-meio : x_int+meio+1]
    recorte = np.nan_to_num(recorte, nan=np.nanmedian(recorte))
    xc_rel, yc_rel = centroid_com(recorte)
    return x_int - meio + xc_rel, y_int - meio + yc_rel

# ==========================================
# 2. LOOP PRINCIPAL POR FILTRO
# ==========================================
for filtro, arquivos in arquivos_por_filtro.items():
    # Ordena os arquivos cronologicamente/alfabeticamente
    arquivos = sorted(arquivos)
    total_arq = len(arquivos)
    
    print(f"{'='*60}")
    print(f"INICIANDO PROCESSAMENTO DO FILTRO: '{filtro.upper()}' ({total_arq} imagens)")
    print(f"{'='*60}")
    
    imagem_referencia = arquivos[0]
    
    # --- A) DETEÇÃO NA 1ª IMAGEM DO FILTRO ---
    print(f"Buscando feixes na imagem de referência: {os.path.basename(imagem_referencia)}...")
    data_ref = fits.getdata(imagem_referencia)
    data_ref = np.squeeze(data_ref)
    data_ref = np.nan_to_num(data_ref, nan=np.nanmedian(data_ref), posinf=0, neginf=0)

    mean, median, std = sigma_clipped_stats(data_ref, sigma=3.0)
    bkg, rms = median, std

    daofind = DAOStarFinder(fwhm=3.0, threshold=3.0 * rms, sharpness_range=(0.0, 2.0), roundness_range=(-2.0, 2.0))
    sources = daofind(data_ref - bkg)

    x = np.array(sources["xcentroid"] if "xcentroid" in sources.colnames else sources["x_centroid"])
    y = np.array(sources["ycentroid"] if "ycentroid" in sources.colnames else sources["y_centroid"])
    fluxo = np.array(sources["flux"])

    idx_principal = np.argmax(fluxo)
    x_prin, y_prin = x[idx_principal], y[idx_principal]

    raio_busca = 41.0
    dr = 8.0
    distancias = np.sqrt((x - x_prin)**2 + (y - y_prin)**2)
    mascara_anel = (distancias >= (raio_busca - dr)) & (distancias <= (raio_busca + dr)) & (distancias > 0.1)
    indices_secundarios = np.where(mascara_anel)[0]

    if len(indices_secundarios) > 0:
        fluxos_no_anel = fluxo[indices_secundarios]
        idx_mais_brilhante_no_anel = np.argmax(fluxos_no_anel)
        idx_sec = indices_secundarios[idx_mais_brilhante_no_anel]
        x_sec, y_sec = x[idx_sec], y[idx_sec]
    else:
        print(f"[ERRO] Deteção radial falhou no filtro {filtro.upper()}. Pulando para o próximo filtro...")
        continue
        
    # --- B) VERIFICAÇÃO HUMANA ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Confirmação: Filtro {filtro.upper()} | {os.path.basename(imagem_referencia)}")

    vmin = np.median(data_ref)
    vmax = np.percentile(data_ref, 99.5)

    ax1.imshow(data_ref, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax1.set_title(f"P: Principal ({x_prin:.1f}, {y_prin:.1f})")
    ax1.add_patch(Circle((x_prin, y_prin), 3, color='cyan', fill=False, ls=':', lw=1))
    ax1.add_patch(Circle((x_prin, y_prin), raio_abertura, color='lime', fill=False, lw=2))
    ax1.set_xlim(x_prin - 25, x_prin + 25)
    ax1.set_ylim(y_prin - 25, y_prin + 25)

    ax2.imshow(data_ref, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax2.set_title(f"S: Secundário ({x_sec:.1f}, {y_sec:.1f})")
    ax2.add_patch(Circle((x_sec, y_sec), 3, color='cyan', fill=False, ls=':', lw=1))
    ax2.add_patch(Circle((x_sec, y_sec), raio_abertura, color='red', fill=False, lw=2))
    ax2.set_xlim(x_sec - 25, x_sec + 25)
    ax2.set_ylim(y_sec - 25, y_sec + 25)

    plt.tight_layout()
    plt.show()

    confirma = input(f"Filtro {filtro.upper()}: As estrelas estão marcadas corretamente? (S para confirmar, N para pular filtro): ").upper()

    if confirma != 'S' and confirma != '':
        print(f" -> Filtro {filtro.upper()} ignorado pelo operador. Pulando para o próximo...\n")
        continue # Pula para o próximo filtro sem travar o código
        
    print(f"\n -> Iniciando extração rápida de {total_arq} imagens no filtro {filtro.upper()}...")

    # --- C) RASTREAMENTO, FOTOMETRIA E SALVAMENTO (.CSV) ---
    arquivo_saida = os.path.join(pasta_obj_reduced, f'fotometria_filtro_{filtro}.csv')

    with open(arquivo_saida, 'w') as f_out:
        # Cabeçalho para o astropop
        f_out.write("Arquivo,Posicao,Angulo,X_Prin,Y_Prin,Fluxo_Prin,X_Sec,Y_Sec,Fluxo_Sec\n")
        
        for i, arq in enumerate(arquivos):
            nome = os.path.basename(arq)
            
            # Lógica de cálculo do ângulo IAGpol
            partes = nome.split('_')
            try:
                posicao_str = partes[3]            
                posicao_int = int(posicao_str)     
                angulo_psi = (posicao_int - 1) * 22.5 
            except (IndexError, ValueError):
                posicao_int = 0
                angulo_psi = 0.0
            
            dados = fits.getdata(arq)
            dados = np.squeeze(dados)
            dados = np.nan_to_num(dados, nan=np.nanmedian(dados), posinf=0, neginf=0)
            
            x_prin, y_prin = rastrear_centroide(dados, x_prin, y_prin)
            x_sec, y_sec = rastrear_centroide(dados, x_sec, y_sec)
            
            aberturas = CircularAperture([(x_prin, y_prin), (x_sec, y_sec)], r=raio_abertura)
            tabela_fotometria = aperture_photometry(dados, aberturas)
            
            fluxo_p = tabela_fotometria['aperture_sum'][0]
            fluxo_s = tabela_fotometria['aperture_sum'][1]
            
            # Imprime na tela a cada 50 imagens para não travar o terminal, ou na última
            if i % 50 == 0 or i == total_arq - 1:
                print(f"[{i+1}/{total_arq}] {nome} (Pos {posicao_int} | {angulo_psi}°) -> F_Prin={fluxo_p:.0f} | F_Sec={fluxo_s:.0f}")
            
            f_out.write(f"{nome},{posicao_int},{angulo_psi:.1f},{x_prin:.3f},{y_prin:.3f},{fluxo_p:.3f},{x_sec:.3f},{y_sec:.3f},{fluxo_s:.3f}\n")

    print(f"\n✅ Filtro {filtro.upper()} finalizado! Dados salvos em: {arquivo_saida}\n")

print("===================================================")
print("=== PIPELINE DE EXTRAÇÃO CONCLUÍDO COM SUCESSO! ===")
print("===================================================")