from Todim import *
import pandas as pd


def main():

    theta = 1     # θ perdas
    pesos = [30., 60., 100., 50., 40.]
    colunas = ['Ativo', 'vpl_por_cota-Jan-2022','taxa_adm-Jan-2022','dy-Jan-2022','patrimonio_liquido-Jan-2022','cotacao-Jan-2022']
    segmentos = []
    tamanho_carteira = 10

    # segmentos = [
    # 'Lajes Corporativas', 'Hotéis', 'Imóveis Industriais e Logísticos',
    # 'Imóveis Comerciais - Outros', 'Imóveis Residenciais', 'Logística', 'Residencial', 'Hotel',
    # 'Educacional', 'Hospitalar','Gás',
    # 'Shoppings', 'Varejo', 'Agências de Bancos',
    # 'Papéis', 'Fundo de Fundos', 'Fundo de Desenvolvimento','Títulos e Valores Mobiliários',
    # 'Indefinido', 'Misto', 'Híbrido']


    # Pré processamento
    (codigos, matriz) = ingestData(colunas, segmentos)

    ################################################
    # TODIM Clássico

    #Carregar a matriz de decisão
    todim = Todim(matriz, pesos, codigos, theta,tamanho_carteira, max=True)

    # Normalizar a matriz de decisão de forma que em cada coluna o valor total seja igual a um.
    todim.normalize_matrix()

    # Normalizar ao peso dos criterios para que a soma de todos os pesos seja igual a um.
    todim.normalize_weights()

    # Calcular do grau de domínio global ζ
    todim.get_grau_dominio()

    # Plotar o gráfico de barras
    todim.plot_bars()


def ingestData(colunas, segmentos, filename='input.xlsx'):
    # print("Importando o .csv")
    if len(colunas) < 1:
        print("Erro nos parametros de {}()".format(ingestData.__name__))
        raise ValueError
    try:

        raw_matrix = pd.read_excel(filename, sheet_name='input')

        # filtra os segmentos
        if segmentos != None and len(segmentos) > 0:
            raw_matrix = raw_matrix[raw_matrix.Segmento.isin(
                segmentos)]
            

        # faz a exclusão das linhas que contem algum valor numerico nulo.
        matrix = raw_matrix[colunas]
        matrix = matrix[(matrix != '-').all(axis=1)]

        # converte tudo para float menos a primeiracoluna
        matrix[matrix.columns[1:]] = matrix[matrix.columns[1:]].astype(float)

        # matrix['vpl_por_cota/cotacao'] = matrix[matrix.columns[1]] / matrix[matrix.columns[2]]

        # #remove outliers
        # q = matrix["vpl_por_cota/cotacao"].quantile(0.99)
        # matrix = matrix[matrix["vpl_por_cota/cotacao"] < q]

        # "codes" são as abreviações de cada fii, elas só serão usadas para mostrar os resultados ao final
        codes = matrix[colunas[:1]]

        # retira o codigo da matrix
        matrix = matrix[colunas[1:]]


    except IOError:
        print("Erro na leitura do arquivo de entrada em ingestCSV()!")
        raise IOError

    return (codes, matrix)

if __name__ == '__main__':
    main()
