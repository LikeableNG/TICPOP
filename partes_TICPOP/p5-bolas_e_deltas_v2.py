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
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# ==========================================
# 1. CONFIGURAÇÃO E ARQUIVOS
# ==========================================
pasta_reduced_estrela = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/osco/'
todos_arquivos = glob.glob(os.path.join(pasta_reduced_estrela, 'red_*.fits'))

if not todos_arquivos:
    raise ValueError("Nenhum arquivo 'red_*.fits' encontrado.")

arquivos_por_filtro = defaultdict(list)
for arq in todos_arquivos:
    nome = os.path.basename(arq)
    partes = nome.split('_')
    if len(partes) >= 5:
        filtro = partes[2]
        arquivos_por_filtro[filtro].append(arq)

print("=== PIPELINE BLINDADO DE FOTOMETRIA BEACON ===")
print(f"Filtros identificados: {list(arquivos_por_filtro.keys())}\n")

# Parâmetros Fotométricos
raio_abertura = 6.0       # Raio da estrela (px)
r_in_sky = 10.0           # Raio interno do anel de céu (px)
r_out_sky = 15.0          # Raio externo do anel de céu (px)
janela_busca = 12         # Janela max de busca ao redor da coordenada âncora (px)

def rastrear_centroide_ancorado(dados, x_ancora, y_ancora, janela=12):
    """
    Rastreia a estrela SEM acumular erro. Procura sempre ao redor da âncora original.
    Subtrai o fundo local para não ser puxado por gradientes de ruído.
    """
    x_int, y_int = int(round(x_ancora)), int(round(y_ancora))
    h, w = dados.shape
    
    y_min, y_max = max(0, y_int - janela), min(h, y_int + janela + 1)
    x_min, x_max = max(0, x_int - janela), min(w, x_int + janela + 1)
    
    recorte = dados[y_min:y_max, x_min:x_max]
    recorte = np.nan_to_num(recorte, nan=np.nanmedian(recorte))
    
    # Subtrai mediana local para o centróide focar no pico da estrela
    recorte_sub = np.maximum(recorte - np.median(recorte), 0)
    xc_rel, yc_rel = centroid_com(recorte_sub)
    
    x_novo = x_min + xc_rel
    y_novo = y_min + yc_rel
    
    # Se o desvio for fora do limite aceitável, segura na posição de âncora
    if np.abs(x_novo - x_ancora) > (janela - 2) or np.abs(y_novo - y_ancora) > (janela - 2):
        return x_ancora, y_ancora
        
    return x_novo, y_novo

# Area das aberturas para subtracao do ceu
area_aper = np.pi * (raio_abertura ** 2)
area_ann = np.pi * (r_out_sky ** 2 - r_in_sky ** 2)

# ==========================================
# 2. LOOP POR FILTRO
# ==========================================
for filtro, arquivos in arquivos_por_filtro.items():
    arquivos = sorted(arquivos)
    total_arq = len(arquivos)
    
    print(f"\n{'='*60}")
    print(f"PROCESSANDO FILTRO: '{filtro.upper()}' ({total_arq} imagens)")
    print(f"{'='*60}")
    
    imagem_referencia = arquivos[0]
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
    x_prin_ref, y_prin_ref = x[idx_principal], y[idx_principal]

    raio_busca = 41.0
    dr = 8.0
    distancias = np.sqrt((x - x_prin_ref)**2 + (y - y_prin_ref)**2)
    mascara_anel = (distancias >= (raio_busca - dr)) & (distancias <= (raio_busca + dr)) & (distancias > 0.1)
    indices_secundarios = np.where(mascara_anel)[0]

    if len(indices_secundarios) > 0:
        fluxos_no_anel = fluxo[indices_secundarios]
        idx_sec = indices_secundarios[np.argmax(fluxos_no_anel)]
        x_sec_ref, y_sec_ref = x[idx_sec], y[idx_sec]
    else:
        print(f"[ERRO] Deteção radial falhou no filtro {filtro.upper()}. Pulando...")
        continue

    # VÍNCULO RÍGIDO ENTRE OS FEIXES (Calcita IAGpol)
    dx_rigido = x_sec_ref - x_prin_ref
    dy_rigido = y_sec_ref - y_prin_ref

    # --- VERIFICAÇÃO HUMANA ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Confirmação: Filtro {filtro.upper()} | Ref: {os.path.basename(imagem_referencia)}")

    vmin = np.median(data_ref)
    vmax = np.percentile(data_ref, 99.5)

    ax1.imshow(data_ref, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax1.set_title(f"P: Principal ({x_prin_ref:.1f}, {y_prin_ref:.1f})")
    ax1.add_patch(Circle((x_prin_ref, y_prin_ref), raio_abertura, color='lime', fill=False, lw=2))
    ax1.add_patch(Circle((x_prin_ref, y_prin_ref), r_in_sky, color='yellow', fill=False, ls=':'))
    ax1.add_patch(Circle((x_prin_ref, y_prin_ref), r_out_sky, color='yellow', fill=False, ls=':'))
    ax1.set_xlim(x_prin_ref - 25, x_prin_ref + 25)
    ax1.set_ylim(y_prin_ref - 25, y_prin_ref + 25)

    ax2.imshow(data_ref, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax2.set_title(f"S: Secundário ({x_sec_ref:.1f}, {y_sec_ref:.1f})")
    ax2.add_patch(Circle((x_sec_ref, y_sec_ref), raio_abertura, color='red', fill=False, lw=2))
    ax2.add_patch(Circle((x_sec_ref, y_sec_ref), r_in_sky, color='yellow', fill=False, ls=':'))
    ax2.add_patch(Circle((x_sec_ref, y_sec_ref), r_out_sky, color='yellow', fill=False, ls=':'))
    ax2.set_xlim(x_sec_ref - 25, x_sec_ref + 25)
    ax2.set_ylim(y_sec_ref - 25, y_sec_ref + 25)

    plt.tight_layout()
    plt.show()

    confirma = input(f"Filtro {filtro.upper()}: As estrelas e anéis de céu estão corretos? (S/N): ").upper()
    if confirma != 'S' and confirma != '':
        continue

    # --- EXTRAÇÃO COM SUBTRAÇÃO DE CÉU E VÍNCULO RÍGIDO ---
    arquivo_saida = os.path.join(pasta_reduced_estrela, f'fotometria_filtro_{filtro}.csv')

    with open(arquivo_saida, 'w') as f_out:
        f_out.write("Arquivo,Posicao,Angulo,X_Prin,Y_Prin,Fluxo_Prin,X_Sec,Y_Sec,Fluxo_Sec\n")
        
        # Variáveis de referência para o Alerta Inteligente (inicializadas com a 1ª imagem)
        fluxo_p_ref = None
        fluxo_s_ref = None
        ignorar_alertas = False # Permite ao operador desativar alertas se a noite estiver muito instável

        for i, arq in enumerate(arquivos):
            nome = os.path.basename(arq)
            partes = nome.split('_')
            try:
                posicao_int = int(partes[3])
                angulo_psi = (posicao_int - 1) * 22.5
            except (IndexError, ValueError):
                posicao_int = 0
                angulo_psi = 0.0
            
            dados = fits.getdata(arq)
            dados = np.squeeze(dados)
            dados = np.nan_to_num(dados, nan=np.nanmedian(dados), posinf=0, neginf=0)
            
            # 1. Rastreia o feixe principal ancorado na referência
            x_prin, y_prin = rastrear_centroide_ancorado(dados, x_prin_ref, y_prin_ref, janela=janela_busca)
            
            # 2. Vínculo rígido do feixe secundário (Calcita IAGpol)
            x_sec = x_prin + dx_rigido
            y_sec = y_prin + dy_rigido
            
            # 3. Fotometria com Anel de Céu
            aper_p = CircularAperture((x_prin, y_prin), r=raio_abertura)
            ann_p = CircularAnnulus((x_prin, y_prin), r_in=r_in_sky, r_out=r_out_sky)
            phot_p = aperture_photometry(dados, [aper_p, ann_p])
            
            aper_s = CircularAperture((x_sec, y_sec), r=raio_abertura)
            ann_s = CircularAnnulus((x_sec, y_sec), r_in=r_in_sky, r_out=r_out_sky)
            phot_s = aperture_photometry(dados, [aper_s, ann_s])
            
            bkg_p_pixel = phot_p['aperture_sum_1'][0] / area_ann
            fluxo_p = phot_p['aperture_sum_0'][0] - (bkg_p_pixel * area_aper)
            
            bkg_s_pixel = phot_s['aperture_sum_1'][0] / area_ann
            fluxo_s = phot_s['aperture_sum_0'][0] - (bkg_s_pixel * area_aper)
            
            # Guarda o fluxo da 1ª imagem como referência do brilho esperado da estrela
            if i == 0:
                fluxo_p_ref = max(fluxo_p, 100.0)
                fluxo_s_ref = max(fluxo_s, 100.0)

            # --- LÓGICA DO ALERTA INTELIGENTE ---
            desvio_posicao = np.sqrt((x_prin - x_prin_ref)**2 + (y_prin - y_prin_ref)**2)
            queda_fluxo_p = fluxo_p < (0.50 * fluxo_p_ref) # Perdeu mais de 50% do brilho
            queda_fluxo_s = fluxo_s < (0.50 * fluxo_s_ref)
            # Dispara o alerta APENAS se o fluxo cair mais de 80% (nuvem densa) ou se a estrela sair da janela (> 8.0 px)
            queda_critica = (fluxo_p < 0.20 * fluxo_p_ref) or (fluxo_s < 0.20 * fluxo_s_ref)
            deslocado_critico = desvio_posicao > 8.0

            ha_anomalia = (queda_critica or deslocado_critico)

            if ha_anomalia and not ignorar_alertas:
                # Identifica o motivo do alerta para mostrar no terminal
                motivos = []
                if deslocado_critico: motivos.append(f"Pulo de Posição ({desvio_posicao:.1f}px)")
                if queda_fluxo_p or queda_fluxo_s: motivos.append("Queda de Sinal (>50% NUVEM/PERDA)")
                motivo_str = " | ".join(motivos)
                
                print(f"\n⚠️  [ALERTA NA IMAGEM {i+1}/{total_arq}] {nome}")
                print(f"   Motivo: {motivo_str}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"⚠️ ALERTA: {motivo_str}\n{nome} [{i+1}/{total_arq}]", color='red', fontweight='bold')
                
                vmin = np.median(dados)
                vmax = np.percentile(dados, 99.5)
                
                # Zoom Principal
                ax1.imshow(dados, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                ax1.add_patch(Circle((x_prin, y_prin), raio_abertura, color='lime', fill=False, lw=2))
                ax1.add_patch(Circle((x_prin, y_prin), r_in_sky, color='yellow', fill=False, ls=':'))
                ax1.add_patch(Circle((x_prin, y_prin), r_out_sky, color='yellow', fill=False, ls=':'))
                ax1.set_xlim(x_prin - 25, x_prin + 25)
                ax1.set_ylim(y_prin - 25, y_prin + 25)
                ax1.set_title(f"Feixe P: Fluxo={fluxo_p:.0f} (Ref: {fluxo_p_ref:.0f})")
                
                # Zoom Secundário
                ax2.imshow(dados, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                ax2.add_patch(Circle((x_sec, y_sec), raio_abertura, color='red', fill=False, lw=2))
                ax2.add_patch(Circle((x_sec, y_sec), r_in_sky, color='yellow', fill=False, ls=':'))
                ax2.add_patch(Circle((x_sec, y_sec), r_out_sky, color='yellow', fill=False, ls=':'))
                ax2.set_xlim(x_sec - 25, x_sec + 25)
                ax2.set_ylim(y_sec - 25, y_sec + 25)
                ax2.set_title(f"Feixe S: Fluxo={fluxo_s:.0f} (Ref: {fluxo_s_ref:.0f})")
                
                plt.tight_layout()
                plt.show()
                
                print("Opções de ação:")
                print(" [ENTER/S] Aceitar e Continuar")
                print(" [R]       Recalibrar Posição Âncora (se a estrela mudou de lugar)")
                print(" [M]       Mudar Referência de Fluxo (se o céu escureceu/mudou transparência)")
                print(" [C]       Continuar sem mais alertas neste filtro")
                
                resp = input("Escolha uma opção (ENTER/S/R/M/C): ").upper().strip()
                
                if resp == 'R':
                    x_prin_ref = float(input(f"Novo X âncora (Atual: {x_prin:.1f}): "))
                    y_prin_ref = float(input(f"Novo Y âncora (Atual: {y_prin:.1f}): "))
                    x_prin, y_prin = rastrear_centroide_ancorado(dados, x_prin_ref, y_prin_ref, janela=janela_busca)
                    x_sec = x_prin + dx_rigido
                    y_sec = y_prin + dy_rigido
                elif resp == 'M':
                    fluxo_p_ref = max(fluxo_p, 100.0)
                    fluxo_s_ref = max(fluxo_s, 100.0)
                    print("-> Novo nível de brilho de referência atualizado!")
                elif resp == 'C':
                    ignorar_alertas = True
                    print("-> Alertas desativados para o restante deste filtro. Processando no máximo...")

            # Exibe progresso silencioso no terminal a cada 50 imagens
            if i % 50 == 0 or i == total_arq - 1:
                print(f"[{i+1}/{total_arq}] {nome} | Pos {posicao_int} ({angulo_psi}°) -> F_Prin={fluxo_p:.0f} | F_Sec={fluxo_s:.0f}")
            
            f_out.write(f"{nome},{posicao_int},{angulo_psi:.1f},{x_prin:.3f},{y_prin:.3f},{fluxo_p:.3f},{x_sec:.3f},{y_sec:.3f},{fluxo_s:.3f}\n")

    print(f"✅ Filtro {filtro.upper()} finalizado e salvo em {arquivo_saida}!")

print("\n=== RE-EXTRAÇÃO CONCLUÍDA COM SUCESSO! ===")