import os
import warnings
from collections import defaultdict
from astropy import units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from astropop.image import imcombine, processing, imarith
from astropop.framedata import read_framedata
from astropop.file_collection import FitsFileGroup

warnings.simplefilter('ignore', category=AstropyWarning)
pasta_calib = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/raw/calib/'
pasta_calib_reduced = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/calib/'

def criar_master_flat(pasta_calib, pasta_calib_reduced):
    main_fg = FitsFileGroup(location=pasta_calib, fits_ext=['.fits'], ext=0)
    flat_files = main_fg.filtered({'OBJECT': 'flat'}).files

    # 1. Agrupamento manual duplo (por Filtro e Preamp limpo)
    arquivos_por_grupo = defaultdict(list)
    for arq in flat_files:
        hdr = fits.getheader(arq)
        if hdr.get('NAXIS') == 2:
            # Extrai e limpa o PREAMP (ex: '5.1x' vira '5')
            preamp_raw = str(hdr.get('PREAMP', 'UNKNOWN'))
            preamp_val = preamp_raw.lower().replace('x', '').split('.')[0]
            
            # Extrai o filtro
            filtro_val = str(hdr.get('FILTER', 'UNKNOWN'))
            
            # Agrupa os dois juntos
            arquivos_por_grupo[(filtro_val, preamp_val)].append(arq)

    # 2. Processa cada grupo separadamente
    for (filtro_val, preamp_val), arquivos_flat_2d in arquivos_por_grupo.items():
        print(f"\n--- Processando Master Flat | Filtro: {filtro_val} | PREAMP: {preamp_val} ({len(arquivos_flat_2d)} arquivos) ---")
        
        # Define sufixo e carrega o Bias correto
        sufixo_preamp = f"_g{preamp_val}" if preamp_val != '1' else ""
        caminho_master_bias = os.path.join(pasta_calib_reduced, f'master_bias{sufixo_preamp}.fits')
        
        if not os.path.exists(caminho_master_bias):
            print(f"ERRO: {caminho_master_bias} não encontrado! Abortando este flat.")
            continue
            
        master_bias = read_framedata(caminho_master_bias)

        flat_frames = [read_framedata(f, unit='adu', use_memmap_backend=True) for f in arquivos_flat_2d]
        gain = float(flat_frames[0].header['GAIN']) * u.electron / u.adu

        for f_idx, frame in enumerate(flat_frames):
            print(f" -> Processando flat [{filtro_val}] {f_idx+1}/{len(flat_frames)}", end='\r')
            processing.gain_correct(frame, gain, inplace=True)
            processing.subtract_bias(frame, master_bias, inplace=True)

        print(f"\nCombinando e normalizando Master Flat ({filtro_val}, PREAMP {preamp_val})...")
        master_flat = imcombine(flat_frames, method='median')
        mean_value = master_flat.mean()
        master_flat = imarith(master_flat, mean_value, '/')

        master_flat.meta['astropop master_type'] = 'flat'
        master_flat.meta['FILTER'] = filtro_val
        master_flat.meta['PREAMP'] = preamp_val

        # Salva com o nome correto
        filtro_minusculo = filtro_val.lower()
        nome_out = f'master_flat{sufixo_preamp}_{filtro_minusculo}.fits'
        
        mflat_path = os.path.join(pasta_calib_reduced, nome_out)
        master_flat.write(mflat_path, overwrite=True, no_fits_standard_units=True)

        print(f"Salvo: {mflat_path}")
        del flat_frames

criar_master_flat(pasta_calib, pasta_calib_reduced)