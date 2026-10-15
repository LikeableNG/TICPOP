import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from astropop.polarimetry import match_pairs

# 1. CARREGAR A IMAGEM
caminho = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/paqr/paqr_r_001_proc.fits'
data = fits.getdata(caminho)
data = np.squeeze(data)
data = np.nan_to_num(data, nan=np.nanmedian(data), posinf=0, neginf=0)

# 2. FUNDO E RUÍDO
mean, median, std = sigma_clipped_stats(data, sigma=3.0)
bkg, rms = median, std

# 3. DETECÇÃO (Threshold mais alto para detectar APENAS as 2 estrelas reais)
daofind = DAOStarFinder(fwhm=5.0, threshold=20.0 * rms)
sources = daofind(data - bkg)

if "xcentroid" in sources.colnames:
    x = np.array(sources["xcentroid"])
    y = np.array(sources["ycentroid"])
else:
    x = np.array(sources["x_centroid"])
    y = np.array(sources["y_centroid"])

print(f"-> Total de estrelas detectadas: {len(x)}")
for i, (xi, yi) in enumerate(zip(x, y)):
    print(f"   Fonte {i+1}: X = {xi:.2f}, Y = {yi:.2f}")

# 4. DEFINIR O DESLOCAMENTO REAL DA CALCITA/PRISMA (Diagonal: dx=29, dy=29)
dx_real = 29.0
dy_real = 29.0

# 5. EMPARELHAMENTO DOS FEIXES
pairs = match_pairs(x, y, dx_real, dy_real, tolerance=5.0)

print(f"-> Total de pares emparelhados: {len(pairs)}")

# 6. PLOT GRÁFICO
plt.figure(figsize=(8, 8))
vmin = np.median(data)
vmax = np.percentile(data, 99.5)
plt.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

plt.plot(x, y, "cyan", marker="+", ls="", ms=8, mew=2, label="Fontes Detectadas")

for pair in pairs:
    idx_o = pair["o"]
    idx_e = pair["e"]
    plt.plot(
        [x[idx_o], x[idx_e]],
        [y[idx_o], y[idx_e]],
        "r-",
        lw=2,
        label="Par Ordinário <-> Extraordinário",
    )

plt.title(f"Emparelhamento Correto: {len(pairs)} par emparelhado")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()