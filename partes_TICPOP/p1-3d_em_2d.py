import os
import glob
import astropy
from astropy.io import fits

def descompactar_cubos_3d(pasta):    
    # Encontra todos os arquivos FITS na pasta de entrada
    arquivos_fits = glob.glob(os.path.join(pasta, '*.fits'))
    print(f"Encontrados {len(arquivos_fits)} arquivos na pasta de entrada.\n")
    
    for caminho_arquivo in arquivos_fits:
        nome_base = os.path.basename(caminho_arquivo).replace('.fits', '')

        # Abre o arquivo extraindo a matriz de dados e o cabeçalho (header)
        dados, header = fits.getdata(caminho_arquivo, header=True)
        
        # Verifica a dimensão da matriz (se é um cubo 3D)
        if dados.ndim == 3:
            num_imagens = dados.shape[0]
            print(f"Descompactando '{nome_base}.fits' ({num_imagens} fatias)...")
            
            for i in range(num_imagens):
                fatia_2d = dados[i]
                
                # Cria um nome sequencial (ex: bias_001_001.fits)
                nome_saida = f"{nome_base}_{i+1:03d}.fits"
                caminho_saida = os.path.join(pasta, nome_saida)
                
                # Salva a fatia 2D com o cabeçalho original. O astropy ajusta 
                # automaticamente as chaves NAXIS do cabeçalho para 2D.
                fits.writeto(caminho_saida, fatia_2d, header, overwrite=True)
                
        elif dados.ndim == 2:
            print(f"Ignorando '{nome_base}.fits' (Já é 2D).")
        else:
            print(f"Ignorando '{nome_base}.fits' (Formato não suportado: {dados.ndim}D).")
            
    print("\nDescompactação concluída!")


pasta_arquivos_compactados = '/home/likeableng/BeACoN/arquivos_de_reducao/25jul19/raw/calib/'


descompactar_cubos_3d(pasta_arquivos_compactados)