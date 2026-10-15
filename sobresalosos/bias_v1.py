import os
import numpy as np
from astropy import units as u
from astropy.io import fits

from astropop.image import imcombine, processing, imarith
from astropop.framedata import read_framedata
from astropop.file_collection import FitsFileGroup

CAMINHO_BEACON = '/home/likeableng/BeACoN/arquivos_de_reducao/'
NOME_RAIZ_ESTRELA = "paqr"
DATA="25jul19"

raw_dir = f'{CAMINHO_BEACON}{DATA}/raw/'
reduced_dir = f'{CAMINHO_BEACON}{DATA}/reduced/'

pasta_calib = f'{raw_dir}/calib/'

pasta_calib_reduced = f'{reduced_dir}/calib/'
os.makedirs(pasta_calib_reduced, exist_ok=True)

print(f'Pasta de calibração: {pasta_calib}')

main_fg = FitsFileGroup(location=pasta_calib, fits_ext=['.fits'], ext=0)
print(f'Total de arquivos carregados: {len(main_fg)}')

bias_fg = main_fg.filtered({'OBJECT': 'bias'})
flat_fg = main_fg.filtered({'OBJECT': 'flat'})

print(f'Arquivos de Bias encontrados: {len(bias_fg)}')
print(f'Arquivos de Flat encontrados: {len(flat_fg)}')



# ==============================================================
# PROCESSO RECOMENDADO ASTROPOP (COM FILTRO PARA IGNORAR OS 3D)
# ==============================================================

# 1. Filtramos a lista verificando FISICAMENTE as dimensões no cabeçalho FITS
arquivos_bias_2d = []
for f in bias_fg.files:
    # Lê apenas o cabeçalho (é instantâneo) e verifica o número de eixos (NAXIS)
    header = fits.getheader(f)
    if header.get('NAXIS') == 2:
        arquivos_bias_2d.append(f)

print(f'Arquivos 2D de Bias purificados para leitura: {len(arquivos_bias_2d)}')

# 2. Carrega apenas os arquivos 2D limpos usando read_framedata
bias_frames = []
for arq in arquivos_bias_2d:
    fd = read_framedata(arq, unit='adu', use_memmap_backend=True)
    bias_frames.append(fd)

# 2. Carrega apenas os arquivos 2D limpos usando read_framedata
bias_frames = []
for arq in arquivos_bias_2d:
    fd = read_framedata(arq, unit='adu', use_memmap_backend=True)
    bias_frames.append(fd)

# 3. Pega o ganho do primeiro arquivo carregado
gain = float(bias_frames[0].header['GAIN']) * u.electron/u.adu

for i, frame in enumerate(bias_frames):
    print(f'Processando bias frame {i+1} de {len(bias_frames)}', end='\r')
    # Remove raios cósmicos e corrige o ganho
    processing.cosmics_lacosmic(frame, inplace=True)
    processing.gain_correct(frame, gain, inplace=True)

print("\nCombinando frames...")
master_bias = imcombine(bias_frames, method='median')
master_bias.meta['astropop master_type'] = 'bias'

mbias_name = os.path.join(pasta_calib_reduced, 'master_bias.fits')
master_bias.write(mbias_name, overwrite=True, no_fits_standard_units=True)
print(f'Master Bias salvo em: {mbias_name}')

del bias_frames