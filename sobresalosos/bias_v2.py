import os
import glob
import warnings
from astropy.io import fits
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from astropop.image import imcombine, processing, imarith
from astropop.framedata import read_framedata
from astropop.file_collection import FitsFileGroup

# Silencia os avisos chatos do HIERARCH
warnings.simplefilter('ignore', category=AstropyWarning)

# ==========================================
# 2. FUNÇÕES DE CALIBRAÇÃO (SEM LA COSMIC)
# ==========================================
def criar_master_bias(pasta_entrada, pasta_saida):
    """Cria o Master Bias focando em velocidade (sem L.A. Cosmic)."""
    print("\n[Calibração] Criando Master Bias...")
    fg = FitsFileGroup(location=pasta_entrada, fits_ext=['.fits'])
    bias_fg = fg.filtered({'OBJECT': 'bias'})
    
    # Filtra só os 2D
    arq_2d = [f for f in bias_fg.files if fits.getheader(f).get('NAXIS') == 2]
    
    frames = [read_framedata(f, unit='adu', use_memmap_backend=True) for f in arq_2d]
    gain = float(frames[0].header['GAIN']) * u.electron/u.adu
    
    for frame in frames:
        processing.gain_correct(frame, gain, inplace=True)
        # NADA de cosmics_lacosmic aqui!
        
    master = imcombine(frames, method='median')
    master.meta['astropop master_type'] = 'bias'
    
    caminho_out = os.path.join(pasta_saida, 'master_bias.fits')
    master.write(caminho_out, overwrite=True, no_fits_standard_units=True)
    return master



# ==========================================
# BLOCO DE EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == '__main__':
    pasta_calib = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/raw/calib/'
    pasta_obj = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/raw/objetos/' # Exemplo
    pasta_out_calib = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/reduced/calib/'
    
    os.makedirs(pasta_out_calib, exist_ok=True)
    
    # 1. Descompacta tudo o que for necessário
    #extrair_cubos_2d(pasta_calib)
    # extrair_cubos_2d(pasta_obj)
    
    # 2. Gera calibrações rápidas
    mbias = criar_master_bias(pasta_calib, pasta_out_calib)
    
    # 3. Reduz os objetos rigorosamente
    # reduzir_objetos(pasta_obj, pasta_out_calib, mbias)