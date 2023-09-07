import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Todim:

    matrix_d = None         # A matriz de decisão com as alternativas e criterios
    maximization = None     # Maximizar ou minimizar?
    weights = None          # Os pesos para cada criterio
    codes = None            # Os códigos de identificação das empresas
    wref = None             # Peso de referencia
    theta = None            # O valor de theta
    n_alt = None            # O numero de alternativas
    n_cri = None            # O numero de criterios
    norm_matrix_d = None    # A matrizD normalizada
    phi = None              # Parcela de contribuição de um criterio
    delta = None            # Matriz n_alt x n_alt, cada endereço é composto pela dominancia de uma alternativa i sobre uma j
    c_proximidade = None    # O coeficiente relativo de proximidade
    tamanho_carteira = None # Tamanho da carteira

    # Etapa 1 - cria a matriz de decisão
    def __init__(self, matriz, pesos, codigos, theta, tamanho_carteira, max=True):
        self.maximization = max
        self.matrix_d = np.asarray(matriz)
        self.weights = np.asarray(pesos)
        self.codes = np.asarray(codigos)
        self.theta = theta
        self.tamanho_carteira = tamanho_carteira

        # inicializar as variaveis
        tam = self.matrix_d.shape
        [self.n_alt, self.n_cri] = tam
        self.norm_matrix_d = np.zeros(tam, dtype=float)
        self.delta = np.zeros([self.n_alt, self.n_alt])
        self.r_closeness = np.zeros([self.n_alt, 1], dtype=float)

    # Etapa 2 - normaliza a matriz de decisão
    def normalize_matrix(self):
        # somatório dos valores de desempenho das alternativas de um criterio (criterios são colunas ( soma das linhas de uma coluna axis = 0))
        m = self.matrix_d.sum(axis=0)
        
        for i in range(self.n_alt):
            for j in range(self.n_cri):
                self.norm_matrix_d[i, j] = self.matrix_d[i, j] / m[j]
        self.matrix_d = self.norm_matrix_d

    # Etapa 3 - normalizar os normaliza os pesos
    def normalize_weights(self):
        if self.weights.sum() > 1.0000001 or self.weights.sum() < 0.9999999:
            # pnc = wc/pr onde pr = sum(wc)
            self.weights = self.weights/self.weights.sum()
        # peso de referencia - wr (o maior peso entre todos os pesos dos critérios normalizados)
        self.wref = self.weights.max()

    # Etapa 5 - ζ - calcula o grau de dominio (matriz dominancia final)
    def get_grau_dominio(self):

        self.get_sum_delta()
        # todos os delta de uma linha (todos os criterios de uma alternativa)
        aux = self.delta.sum(axis=1)
        for i in range(self.n_alt):
            self.r_closeness[i] = (aux[i] - aux.min()) / \
                (aux.max() - aux.min())

    # Etrapa 4 - δ - (Ai,Aj)
    def get_sum_delta(self):
        # somatorio de todos os deltas
        for i in range(self.n_alt):
            for j in range(self.n_alt):
                self.delta[i, j] = self.get_sum_phi(i, j)

    def get_sum_phi(self, i, j):
        m = 0
        for c in range(self.n_cri):
            m = m + self.get_phi(i, j, c)
        return m
    
    # Φ phi
    def get_phi(self, i, j, c):
        # etapa 3 - adota-se wrc como representação da taxa de substituição do critério c em relação ao critério de referencia wr
        # wrc = pnc/wr
        wcr = self.weights[c]/self.wref
        # -----------------

        # somatório das taxas de substituição (∑mc=1 wrc)
        sum_w_ref = self.get_sum_w_ref()
        
        # (Pic −Pjc)
        dij = self.get_distance(i, j, c)

        comp = self.get_comparison(i, j, c)
        
        if comp == 0:
            return 0
        elif comp > 0:
            return np.sqrt((wcr*abs(dij))/sum_w_ref)
        else:
            return np.sqrt((sum_w_ref*abs(dij))/wcr)/(-self.theta)

    def get_sum_w_ref(self):
        sum_w_ref = 0
        for c in self.weights:
            sum_w_ref += c / self.wref
        return sum_w_ref

    def get_distance(self, alt_i, alt_j, crit):
        return (self.matrix_d[alt_i, crit] - self.matrix_d[alt_j, crit])

    # funcao modular para possibilitar outros tipos de comparações
    def get_comparison(self, alt_i, alt_j, crit):
        return self.get_distance(alt_i, alt_j, crit)
    
    def plot_bars(self, names=None, save_name=None):

        # une as matrizes de codigos e de grau de dominancia
        all_data = np.append(self.codes, self.r_closeness, 1)

        # ordena a matriz numpy de acordo com a segunda coluna(valores finais obtidos pelo Todim)
        # fica apenas com os maiores valores para compor a carteira
        all_data = all_data[all_data[:, 1].argsort()[::-1]][:self.tamanho_carteira]
        # exibe uma grafico de barras
        sns.set_style("whitegrid")

        # coloca margens no grafico
        sns.set(rc={'figure.figsize': (8, 5)})

        
        a = sns.barplot(x=all_data[:, 1], y=all_data[:, 0], palette="summer")

        a.set_xlabel("Grau de Dominância")
        a.set_ylabel('Alternativas')
        fig = a.get_figure()
        plt.title('Ranking')
        plt.show()

        if save_name is not None:
            fig.savefig(save_name+'.png')
