import numpy as np
import matplotlib.pyplot as plt

class Todim:
    def __init__(self, matriz, pesos, codigos, tamanho_carteira=5, theta=1, maximiza=None, debug=False):
        self.matrix_d = np.asarray(matriz, dtype=float)
        self.weights = np.asarray(pesos, dtype=float)
        self.codes = np.asarray(codigos).reshape(-1, 1)
        self.tamanho_carteira = tamanho_carteira
        self.theta = theta
        self.debug = debug

        assert self.matrix_d.shape[0] == len(codigos), "Número de códigos não bate com número de alternativas"
        assert self.matrix_d.shape[1] == len(pesos), "Número de pesos deve ser igual ao número de critérios"
        assert self.tamanho_carteira <= self.matrix_d.shape[0], "Carteira maior que número de alternativas"

        # Se maximiza não foi passado, assume todos True (maximiza todos)
        if maximiza is None:
            self.maximiza = np.array([True] * self.matrix_d.shape[1])
        else:
            if len(maximiza) != self.matrix_d.shape[1]:
                raise ValueError("O vetor maximiza deve ter o mesmo tamanho que o número de critérios.")
            self.maximiza = np.array(maximiza, dtype=bool)

        self.norm_matrix_d = None
        self.norm_weights = None
        self.wref = None
        self.r_closeness = None

    def normalize_matrix(self):
        # Normaliza por soma da coluna
        self.norm_matrix_d = self.matrix_d / self.matrix_d.sum(axis=0)

        # Inverte os valores dos critérios que são minimizados
        for i, maximiza in enumerate(self.maximiza):
            if not maximiza:
                self.norm_matrix_d[:, i] = 1 - self.norm_matrix_d[:, i]

        if self.debug:
            print("Matriz normalizada e ajustada (maximiza):\n", self.norm_matrix_d)
        return self

    def normalize_weights(self):
        self.norm_weights = self.weights / self.weights.sum()
        self.wref = np.max(self.norm_weights)
        if self.debug:
            print("Pesos normalizados:", self.norm_weights)
            print("Peso de referência:", self.wref)
        return self

    def get_distance(self, i, j, c):
        return self.norm_matrix_d[i, c] - self.norm_matrix_d[j, c]

    def get_relative_weight(self, c):
        return self.norm_weights[c] / self.wref

    def dominance(self, dij, wr):
        if dij == 0:
            return 0
        if dij > 0:
            return np.sqrt(wr * dij)
        return -np.sqrt(abs(dij * wr) / self.theta)

    def compute_dominance(self):
        n_alt = self.norm_matrix_d.shape[0]
        delta = np.zeros((n_alt, n_alt))

        for i in range(n_alt):
            for j in range(n_alt):
                soma = 0
                for c in range(len(self.norm_weights)):
                    dij = self.get_distance(i, j, c)
                    wr = self.get_relative_weight(c)
                    soma += self.dominance(dij, wr)
                delta[i, j] = soma

        phi = delta.sum(axis=1)
        self.r_closeness = (phi - np.min(phi)) / (np.max(phi) - np.min(phi))
        self.r_closeness = self.r_closeness.reshape(-1, 1)
        if self.debug:
            print("Valores de closeness:\n", self.r_closeness)
        return self

    def get_ranked_alternatives(self):
        data = np.append(self.codes, self.r_closeness, axis=1)
        return data[data[:, 1].astype(float).argsort()[::-1]]

    def plot_bars(self):
        ranked = self.get_ranked_alternatives()[:self.tamanho_carteira]
        nomes = ranked[:, 0]
        valores = ranked[:, 1].astype(float)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(nomes, valores, color='steelblue')
        plt.title('Alternativas Selecionadas - Método TODIM')
        plt.xlabel('Alternativas')
        plt.ylabel('Closeness')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar, value in zip(bars, valores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    def run(self):
        return self.normalize_matrix().normalize_weights().compute_dominance()
