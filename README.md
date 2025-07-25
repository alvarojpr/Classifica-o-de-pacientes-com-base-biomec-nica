# Classificação de Pacientes com Base em Atributos Biomecânicos

## Dataset

Utilizou-se dois datasets do UCI:
- `column_2C_weka.csv`: classificação binária (normal vs anormal)
- `column_3C_weka.csv`: classificação em 3 classes (normal, hérnia, espondilolistese)

## Etapas:

1. Verificação e remoção de valores ausentes;
2. Divisão aleatória entre treino e teste;
3. Treinamento com 3 classificadores:
   - Árvore de Decisão
   - Naive Bayes
   - SVM
4. Avaliação com métricas de acurácia;
5. Geração de matrizes de confusão e comparativo entre os modelos.

Bibliotecas: `pandas`, `sklearn`, `matplotlib`, `seaborn`
