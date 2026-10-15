import os
import warnings
from collections import defaultdict
from astropy.io import fits
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from astropop.image import imcombine, processing
from astropop.framedata import read_framedata
from astropop.file_collection import FitsFileGroup

warnings.simplefilter('ignore', category=AstropyWarning)

def criar_master_bias(pasta_entrada, pasta_saida):
    print("\n[Calibração] Criando Master Bias por PREAMP...")
    fg = FitsFileGroup(location=pasta_entrada, fits_ext=['.fits'], ext=0)
    
    bias_files = fg.filtered({'OBJECT': 'bias'}).files
    
    arquivos_por_preamp = defaultdict(list)
    for arq in bias_files:
        hdr = fits.getheader(arq)
        if hdr.get('NAXIS') == 2:
            # Pega o valor original (ex: '5.1x' ou '1x')
            preamp_raw = str(hdr.get('PREAMP', 'UNKNOWN'))
            
            # Limpa o texto: tira o 'x' e corta tudo depois do ponto
            preamp_val = preamp_raw.lower().replace('x', '').split('.')[0]
            
            arquivos_por_preamp[preamp_val].append(arq)
            
    for preamp_val, arq_2d in arquivos_por_preamp.items():
        print(f"-> Processando Bias para PREAMP: {preamp_val} ({len(arq_2d)} arquivos)")
        
        frames = [read_framedata(f, unit='adu', use_memmap_backend=True) for f in arq_2d]
        gain = float(frames[0].header['GAIN']) * u.electron/u.adu
        
        for frame in frames:
            processing.gain_correct(frame, gain, inplace=True)
            
        master = imcombine(frames, method='median')
        master.meta['astropop master_type'] = 'bias'
        master.meta['PREAMP'] = preamp_val
        
        # Como limpamos o valor, agora ele é exatamente '1' ou '5'
        sufixo_preamp = f"_g{preamp_val}" if preamp_val != '1' else ""
        caminho_out = os.path.join(pasta_saida, f'master_bias{sufixo_preamp}.fits')
        
        master.write(caminho_out, overwrite=True, no_fits_standard_units=True)
        print(f"   Salvo: {caminho_out}\n")

pasta_calib = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/raw/calib/'
pasta_out_calib = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/calib/'

criar_master_bias(pasta_calib, pasta_out_calib)