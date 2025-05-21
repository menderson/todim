# Todim

O método **TODIM** é uma técnica de apoio à decisão multicritério baseada na **Teoria dos Prospectos** ( *Prospect Theory* ), desenvolvida por  **Kahneman e Tversky** . Essa teoria modela como as pessoas tomam decisões em condições de risco e incerteza, levando em conta que os indivíduos avaliam ganhos e perdas de forma  **assimétrica** , ou seja, a  **perda pesa mais do que o ganho de mesma magnitude** .

---

## 🧠 Fundamentos do TODIM

O TODIM parte de uma  **matriz de decisão** , onde cada linha representa uma alternativa (ex: clientes, produtos, projetos etc.), e cada coluna representa um critério (ex: lucro, risco, satisfação etc.).

### 1. **Normalização da Matriz**

Para que critérios de escalas diferentes possam ser comparados, os dados são  **normalizados** . No TODIM clássico, isso é feito pela soma das colunas (critérios), transformando os valores brutos em proporções.

### 2. **Normalização dos Pesos**

Os pesos de cada critério (importância relativa) são normalizados para que sua soma seja 1. O maior peso é escolhido como **referência** (`wref`), sendo usado no cálculo da dominância.

---

## ⚖️ Teoria dos Prospectos no TODIM

A essência do TODIM é o  **modelo comportamental** , onde se avalia como uma alternativa **i** se comporta em relação a outra alternativa  **j** , critério por critério.

* Se a alternativa **i** for **melhor** que **j** no critério  **c** , isso é considerado um  **ganho** .
* Se for  **pior** , isso é considerado uma  **perda** , e é penalizado mais fortemente de acordo com o parâmetro  **θ (theta)** , que representa o nível de  **aversão à perda** .

---

## 🧮 Etapas Computacionais do Método

### 1. **Matriz de decisão**

Contém as alternativas (linhas) e critérios (colunas).

### 2. **Normalização da matriz**

Para tornar os critérios comparáveis entre si.

### 3. **Normalização dos pesos**

Garante proporcionalidade e determina o critério de referência.

### 4. **Cálculo da matriz de dominância**

Calcula o quanto uma alternativa domina a outra (`δ(i,j) = Σφ(i,j,c)` para todos os critérios c).

### 5. **Grau de dominância final**

É somado o total de dominâncias de cada alternativa sobre todas as outras. Depois, os valores são normalizados entre 0 e 1 (coeficiente de proximidade).

---

## 📊 Resultado

As alternativas são **ranqueadas** com base no  **grau de dominância** : quanto maior esse valor, **mais dominante** é a alternativa sobre as demais.

---

## 🎯 Quando usar o TODIM?

* Quando a **percepção do decisor sobre ganhos e perdas** é importante (decisão subjetiva).
* Quando há múltiplos critérios e é desejável considerar a **comportamentalidade humana** (racionalidade limitada, aversão à perda).
* Em problemas como:
  * Seleção de investimentos
  * Escolha de fornecedores
  * Decisão sobre carteiras de clientes
  * Avaliação de desempenho
