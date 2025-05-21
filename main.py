import pandas as pd
from Todim import Todim


def main():
    theta = 1
    tamanho_carteira = 10
    filename = 'input1.xlsx'

    criterios = [
        {"coluna": "cliente"},
        {"peso": 10, "coluna": "ultimo_relacionamento", "maximiza": True},
        {"peso": 10, "coluna": "aniversario_de_cliente", "maximiza": True},
        {"peso": 10, "coluna": "data_da_proxima_agenda", "maximiza": True},
        {"peso": 20,  "coluna": "data_da_ultima_sugestao", "maximiza": True},
        {"peso": 10, "coluna": "saldo_em_conta", "maximiza": True},
        {"peso": 20, "coluna": "vencimento_rf", "maximiza": True},
        {"peso": 20,  "coluna": "oportunidades", "maximiza": True}
    ]

    colunas, pesos, maximiza = extrair_criterios(criterios)

    codigos, matriz = carregar_dados(filename, colunas)

    Todim(
        matriz=matriz,
        pesos=pesos,
        codigos=codigos,
        theta=theta,
        tamanho_carteira=tamanho_carteira,
        maximiza=maximiza,
        debug=False
    ).run().plot_bars()


def extrair_criterios(lista_criterios):
    colunas = [item["coluna"] for item in lista_criterios]
    pesos = [item.get("peso", 0) for item in lista_criterios[1:]]  # Ignora a coluna identificadora, assume peso 0 se não existir
    maximiza = [item.get("maximiza", True) for item in lista_criterios[1:]]  # Assume maximiza True se não definido

    return colunas, pesos, maximiza


def carregar_dados(arquivo, colunas):
    if len(colunas) < 2:
        raise ValueError("É necessário ao menos uma coluna de identificação e um critério numérico.")

    try:
        df = pd.read_excel(arquivo, sheet_name='input')
        df = df[colunas]
        df = df[(df != '-').all(axis=1)]
        df[df.columns[1:]] = df[df.columns[1:]].astype(float)

        codigos = df[[colunas[0]]]  # Ex: cliente
        matriz = df[colunas[1:]]    # Restante das colunas numéricas

        return codigos, matriz

    except Exception as e:
        raise IOError(f"Erro ao ler o arquivo '{arquivo}': {e}")


if __name__ == '__main__':
    main()
