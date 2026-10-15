import os
import glob
import warnings
from astropy import units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from astropop.image import processing
from astropop.framedata import read_framedata
from astropop.file_collection import FitsFileGroup

# Silencia os avisos HIERARCH
warnings.simplefilter('ignore', category=AstropyWarning)

pasta_obj = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/raw/osco/' 
pasta_calib_reduced = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/calib/'
pasta_obj_reduced = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/objetos/'

def reduzir_objetos(pasta_obj, pasta_calib_reduced, pasta_obj_reduced):
    os.makedirs(pasta_obj_reduced, exist_ok=True)

    caminho_mbias = os.path.join(pasta_calib_reduced, 'master_bias.fits')
    print(f"Carregando Master Bias de: {caminho_mbias}")
    master_bias = read_framedata(caminho_mbias)

    print("Carregando Master Flats...")
    master_flats = {}
    arquivos_flats = glob.glob(os.path.join(pasta_calib_reduced, 'master_flat_*.fits'))

    for arq in arquivos_flats:
        filtro = os.path.basename(arq).replace('master_flat_', '').replace('.fits', '')
        master_flats[filtro] = read_framedata(arq)

    print(f"Flats disponíveis carregados para os filtros: {list(master_flats.keys())}")        

    main_fg = FitsFileGroup(location=pasta_obj, fits_ext=['.fits'], ext=0)
    arquivos_obj_2d = [f for f in main_fg.files if fits.getheader(f).get('NAXIS') == 2]
    print(f"Arquivos de objetos 2D encontrados: {len(arquivos_obj_2d)}")

    for i, arq in enumerate(arquivos_obj_2d):
        nome_arq = os.path.basename(arq)
        print(f" -> [{i+1}/{len(arquivos_obj_2d)}] Reduzindo {nome_arq}...", end='\r')

        frame = read_framedata(arq, unit='adu', use_memmap_backend=True)
        gain = float(frame.header['GAIN']) * u.electron / u.adu

        # Identifica o filtro do objeto pelo nome do arquivo (ex: paqr_b_001.fits -> 'b' ou 'B')
        partes = nome_arq.split('_')
        if len(partes) >= 2:
            filtro_obj = partes[1] # Pegando a letra exata para casar com o flat
        else:
            filtro_obj = 'UNKNOWN'

        if filtro_obj not in master_flats:
            print(f"\n[AVISO] Master Flat para o filtro '{filtro_obj}' não encontrado. A saltar {nome_arq}.")
            continue

        processing.cosmics_lacosmic(frame, inplace=True)
        processing.gain_correct(frame, gain, inplace=True)
        processing.subtract_bias(frame, master_bias, inplace=True)
        processing.flat_correct(frame, master_flats[filtro_obj], inplace=True)

        frame.meta['REDUCED'] = True

        nome_saida = f"red_{nome_arq}"
        caminho_saida = os.path.join(pasta_obj_reduced, nome_saida)
        frame.write(caminho_saida, overwrite=True, no_fits_standard_units=True)

    print("\n\nRedução dos objetos concluída com sucesso!")

reduzir_objetos(pasta_obj, pasta_calib_reduced, pasta_obj_reduced)    