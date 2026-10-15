import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from matplotlib.patches import Circle

#CARREGAR A IMAGEM
caminho = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/osco/red_osco_i_009_020.fits'
data = fits.getdata(caminho)
data = np.squeeze(data)
data = np.nan_to_num(data, nan=np.nanmedian(data), posinf=0, neginf=0)

# 2. FUNDO E RUÍDO
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
bkg, rms = median, std

# 3. DETEÇÃO (FWHM um pouco maior para não rejeitar estrelas brilhantes)
daofind = DAOStarFinder(
    fwhm=3.0, 
    threshold=3.0 * rms,
    sharpness_range=(0.0, 2.0),   # Aceita estrelas muito desfocadas ou saturadas
    roundness_range=(-2.0, 2.0)   # Aceita estrelas tortas/elípticas
)
sources = daofind(data - bkg)

# Extrair posições e fluxo (brilho)
x = np.array(sources["xcentroid"] if "xcentroid" in sources.colnames else sources["x_centroid"])
y = np.array(sources["ycentroid"] if "ycentroid" in sources.colnames else sources["y_centroid"])
fluxo = np.array(sources["flux"])

# 4. LÓGICA DE BUSCA RADIAL MELHORADA
# a) Identificar o brilho mais intenso (Feixe Principal)
idx_principal = np.argmax(fluxo)
x0, y0 = x[idx_principal], y[idx_principal]
print(f"-> Feixe Principal (Mais brilhante): X = {x0:.2f}, Y = {y0:.2f}")

# b) Definir os parâmetros da circunferência
raio_busca = 41.0
dr = 8.0  # Tolerância um pouco maior por segurança

# c) Calcular a distância de todas as fontes
distancias = np.sqrt((x - x0)**2 + (y - y0)**2)

# d) Encontrar quais estrelas estão dentro do anel
mascara_anel = (distancias >= (raio_busca - dr)) & (distancias <= (raio_busca + dr)) & (distancias > 0.1)
indices_secundarios = np.where(mascara_anel)[0]

# 5. PLOT GRÁFICO DA BUSCA
plt.figure(figsize=(10, 10))
vmin = np.median(data)
vmax = np.percentile(data, 99.5)
ax = plt.gca()
ax.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

plt.plot(x, y, "cyan", marker="+", ls="", ms=5, label="Fontes Detectadas")
plt.plot(x0, y0, "lime", marker="o", ls="", ms=8, mfc="none", mew=2, label="Feixe Principal")

anel_interno = Circle((x0, y0), raio_busca - dr, color='orange', fill=False, ls='--', lw=1.5, alpha=0.7)
anel_externo = Circle((x0, y0), raio_busca + dr, color='orange', fill=False, ls='--', lw=1.5, alpha=0.7, label=f"Anel (R={raio_busca}±{dr})")
ax.add_patch(anel_interno)
ax.add_patch(anel_externo)

# NOVA LÓGICA: SE ENCONTRAR VÁRIAS COISAS NO ANEL, PEGA A MAIS BRILHANTE
if len(indices_secundarios) > 0:
    # Isola apenas os fluxos das fontes que estão dentro do anel
    fluxos_no_anel = fluxo[indices_secundarios]
    
    # Encontra o índice da mais brilhante DENTRO do anel
    idx_mais_brilhante_no_anel = np.argmax(fluxos_no_anel)
    
    # Recupera o índice real dessa fonte na lista original
    idx_sec = indices_secundarios[idx_mais_brilhante_no_anel]
    
    x1, y1 = x[idx_sec], y[idx_sec]
    print(f"-> Feixe Secundário encontrado: X = {x1:.2f}, Y = {y1:.2f} (Distância: {distancias[idx_sec]:.2f} px)")
    
    plt.plot([x0, x1], [y0, y1], "r-", lw=2, label="Par Emparelhado")
    plt.plot(x1, y1, "red", marker="o", ls="", ms=8, mfc="none", mew=2)
else:
    print("-> Nenhum feixe secundário encontrado dentro do anel.")

plt.title("Emparelhamento por Busca Radial Baseada em Brilho")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()