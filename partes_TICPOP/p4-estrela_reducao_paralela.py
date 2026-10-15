import os
import glob
import warnings
import concurrent.futures
from collections import defaultdict
from astropy import units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

from astropop.image import processing
from astropop.framedata import read_framedata
from astropop.file_collection import FitsFileGroup

warnings.simplefilter('ignore', category=AstropyWarning)

pasta_obj = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/raw/osco/' 
pasta_calib_reduced = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/calib/'
pasta_obj_reduced = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/osco/'

def dividir_em_lotes(lista, tamanho):
    return [lista[i:i + tamanho] for i in range(0, len(lista), tamanho)]

def processar_lote_por_filtro(filtro, preamp, arquivos, caminho_mbias, caminho_mflat, pasta_saida):
    master_bias = read_framedata(caminho_mbias)
    master_flat = read_framedata(caminho_mflat)
    
    total = len(arquivos)
    for i, arq in enumerate(arquivos):
        nome_arq = os.path.basename(arq)
        
        frame = read_framedata(arq, unit='adu', use_memmap_backend=True)
        gain = float(frame.header['GAIN']) * u.electron / u.adu

        processing.cosmics_lacosmic(frame, inplace=True)
        processing.gain_correct(frame, gain, inplace=True)
        processing.subtract_bias(frame, master_bias, inplace=True)
        processing.flat_correct(frame, master_flat, inplace=True)

        frame.meta['REDUCED'] = True
        caminho_saida = os.path.join(pasta_saida, f"{nome_arq}")
        frame.write(caminho_saida, overwrite=True, no_fits_standard_units=True)
        
    return f"Lote [Filtro {filtro} | PREAMP {preamp}] ({total} imagens) CONCLUÍDO!"

def reduzir_objetos_paralelo(pasta_obj, pasta_calib_reduced, pasta_obj_reduced):
    os.makedirs(pasta_obj_reduced, exist_ok=True)

    main_fg = FitsFileGroup(location=pasta_obj, fits_ext=['.fits'], ext=0)
    grupos_objetos = main_fg.group_by(['FILTER', 'PREAMP'])
    
    print("Iniciando processamento paralelo em pequenos lotes seguros...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futuros = []
        
        for chaves, grupo in grupos_objetos:
            filtro_val, preamp_val = chaves
            arquivos_2d = [f for f in grupo.files if fits.getheader(f).get('NAXIS') == 2]
            
            if not arquivos_2d:
                continue
            
            # AQUI: Lógica do sufixo g5 vs sem sufixo
            sufixo_preamp = f"_g{preamp_val}" if str(preamp_val) != '1' else ""

            valor_filtro = filtro_val[1:]
            numero_filtro = int(valor_filtro)

            
            
            caminho_mbias = os.path.join(pasta_calib_reduced, f'master_bias{sufixo_preamp}.fits')
            caminho_mflat = os.path.join(pasta_calib_reduced, f'master_flat{sufixo_preamp}_{letra_filtro[numero_filtro]}.fits')
            
            if not os.path.exists(caminho_mbias) or not os.path.exists(caminho_mflat):
                print(f"[AVISO] Calibração (Bias ou Flat) ausente para Filtro {filtro_val}, PREAMP {preamp_val}. Pulando conjunto.")
                continue
            
            lotes = dividir_em_lotes(arquivos_2d, 50)
            
            for lote in lotes:
                futuro = executor.submit(processar_lote_por_filtro, filtro_val, preamp_val, lote, caminho_mbias, caminho_mflat, pasta_obj_reduced)
                futuros.append(futuro)

        for futuro in concurrent.futures.as_completed(futuros):
            print(futuro.result())

    print("\n\nRedução de TODOS os objetos concluída com sucesso!")

if __name__ == '__main__':
    reduzir_objetos_paralelo(pasta_obj, pasta_calib_reduced, pasta_obj_reduced)
