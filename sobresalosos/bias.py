import glob
import numpy as np
import astropy.io.fits as fits

NOME_RAIZ_ESTRELA = "paqr"
DATA="25jul19"


CAMINHO_BEACON = '/home/likeableng/BeACoN/arquivos_de_reducao/'
PASTA_CALIB = f'{CAMINHO_BEACON}{DATA}/calib/'
PASTA_ESTRELA = f'{CAMINHO_BEACON}{DATA}/{NOME_RAIZ_ESTRELA}/'


arquivos_bias = glob.glob(f'{PASTA_CALIB}*bias_*.fits')

lista_bias = []

for arquivo in arquivos_bias:
    dados_tensor_bias = fits.getdata(arquivo)
    lista_bias.append(dados_tensor_bias)

super_tensor_bias = np.concatenate(lista_bias, axis=0)

master_bias = np.median(super_tensor_bias, axis=0)

fits.writeto(f'{PASTA_CALIB}master_bias.fits', master_bias, overwrite=True)

print(f"-> Master Bias criado com sucesso: {PASTA_CALIB}master_bias.fits")