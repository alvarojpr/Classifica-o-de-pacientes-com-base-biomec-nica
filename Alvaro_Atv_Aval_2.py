from sklearn.tree import DecisionTreeClassifier #aprendizagem por arvore de decisao
from sklearn.naive_bayes import GaussianNB #aprendizagem por metodo bayesiano
from sklearn.svm import SVC #aprendizagem por vetores suport vector machine (SVM)
import pandas as pd # manipular dados
from sklearn.preprocessing import MinMaxScaler # Normalização de atributos dos dados
from sklearn.model_selection import train_test_split #divisão dos dados em conjuntos de treinamento e teste
from sklearn.metrics import confusion_matrix # Matriz de confusão
import seaborn as sns # Biblioteca para visualização de dados baseada em Matplotlib
import matplotlib.pyplot as plt # Importa a biblioteca Matplotlib para visualização de dados

# Carrengando dados nas variáveis
dados1 = pd.read_csv('./column_2C_weka.csv')
dados2 = pd.read_csv('./column_3C_weka.csv')

# Removendo instâncias com valores ausentes.

dados1 = dados1.dropna()
dados2 = dados2.dropna()

scaler = MinMaxScaler() #MinMaxScaler: transforma os dados para que fiquem no intervalo [0, 1]

x = pd.DataFrame(dados1.iloc[:, :-1]) # x recebe todas as linhas e todas as colunas exceto a última dos dados 1
dados1_normalizados = pd.DataFrame(scaler.fit_transform(x), columns=x.columns) #fit_transform() é usado para normalizar os dados, columns=x.columns para manter os nomes das colunas
X1_train, X1_test, y1_train, y1_test = train_test_split(dados1_normalizados, dados1['class'], test_size=15, random_state=42)
# X se refere as linhas, y as colunas.
# train_test_split(Dados normalizados, nome da classe no csv, test_size=10 são os dados de teste), random_state explicado laaaaaaaaá em baixo

x = pd.DataFrame(dados2.iloc[:, :-1]) 
dados2_normalizados = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
X2_train, X2_test, y2_train, y2_test = train_test_split(dados2_normalizados, dados2['class'], test_size=15, random_state=16)

print(f"DADOS NORMALIZADOS 1")
print(dados1_normalizados)
print("\nDADOS NORMALIZADOS 2")
print(dados2_normalizados)

# Aqui eu queria saber quais variaveis de teste estavam sendo usadas
print(f'VARIÁVEIS DE TESTE 1:')
print(f'',X1_test,'\n',y1_test,'\n')
print(f'VARIÁVEIS DE TESTE 2:')
print(f'',X2_test,'\n',y2_test,'\n')

def aprendizagem(Xtrain, Xtest, ytrain, ytest): # optei por uma função para não repetir o código
    print("\nAPRENDIZAGEM")
    # Árvore de decisão 
    dtc = DecisionTreeClassifier(random_state=0) #random_state lá em baixo
    dtc.fit(Xtrain, ytrain) #.fit() é usado para treinar o modelo
    print("Árvore de decisão: ", dtc.score(Xtest, ytest)) # .score() é usado para calcular a precisão do modelo

    # Método bayesiano
    gnb = GaussianNB()
    gnb.fit(Xtrain, ytrain)
    print("Método bayesiano: ", gnb.score(Xtest, ytest))

    # Método de vetores suport vector machine (SVM)
    svm = SVC()
    svm.fit(Xtrain, ytrain)
    print("Método de vetores SVM: ", svm.score(Xtest, ytest))

    return dtc, gnb, svm

# chamada de função	passando os dados de treino e teste
print("\n DADOS 1")
dtc1, gnb1, svm1 = aprendizagem(X1_train, X1_test, y1_train, y1_test)
print("\n DADOS 2")
dtc2, gnb2, svm2 = aprendizagem(X2_train, X2_test, y2_train, y2_test)

def teste_classificacao(Xtest, dtc, gnb, svm): 
    print("\nCLASSIFICAÇÃO 10 INSTÂNCIAS TESTE")
    print("Árvore de decisão: ", dtc.predict(Xtest)) # predict é usado para prever a classe de instâncias
    print("Método bayesiano: ", gnb.predict(Xtest))
    print("Método de vetores SVM: ", svm.predict(Xtest))

print("\n DADOS 1")
teste_classificacao(X1_test, dtc1, gnb1, svm1)
print("\n DADOS 2")
teste_classificacao(X2_test, dtc2, gnb2, svm2)

# a função precisao() mede o score de cada método de classificação
def precisao(X_test, y_test, dtc, gnb, svm): # mede o score 
    print("\nPRECISÃO DA CLASSIFICAÇÃO")
    print("Árvore de decisão:   ", dtc.score(X_test, y_test))
    print("Método bayesiano:       ", gnb.score(X_test, y_test))
    print("Método de vetores SVM: ", svm.score(X_test, y_test))


print("\n DADOS 1")
precisao( X1_test, y1_test, dtc1, gnb1, svm1)
print("\n DADOS 2")
precisao( X2_test, y2_test, dtc2, gnb2, svm2)

# rodando o código, nota-se que para os dados 1 o método de vetores SVM foi o mais preciso, para os dados 2 a arvore de decisão alcansou 
# uma preisão mais alta.

# Função para plotar a matriz de confusão
def plot_confusion_matrix(matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Rotulos previstos')
    plt.ylabel('Rótulos verdadeiros')
    plt.title('Matriz de confusão')
    plt.show()

# Função para calcular a matriz de confusão
def calcular_matriz_confusao(X_test, y_test, dtc, gnb, svm):
    print("\nMATRIZ DE CONFUSÃO")
    print("Árvore de decisão:")
    dtc_pred = dtc.predict(X_test)
    dtc_matrix = confusion_matrix(y_test, dtc_pred)
    print(dtc_matrix)
    plot_confusion_matrix(dtc_matrix, dtc.classes_)
    
    print("\nMétodo bayesiano:")
    gnb_pred = gnb.predict(X_test)
    gnb_matrix = confusion_matrix(y_test, gnb_pred)
    print(gnb_matrix)
    plot_confusion_matrix(gnb_matrix, gnb.classes_)

    print("\nMétodo de vetores SVM:")
    svm_pred = svm.predict(X_test)
    svm_matrix = confusion_matrix(y_test, svm_pred)
    print(svm_matrix)
    plot_confusion_matrix(svm_matrix, svm.classes_)

# Chamada da função para calcular e plotar a matriz de confusão
print("\n Matriz DADOS 1")
calcular_matriz_confusao(X1_test, y1_test, dtc1, gnb1, svm1)

print("\n Matriz DADOS 2")
calcular_matriz_confusao(X2_test, y2_test, dtc2, gnb2, svm2)





# Explicação do random_state:

# O parâmetro random_state é usado em várias funções em bibliotecas como scikit-learn para controlar a aleatoriedade durante a divisão de dados ou durante 
# a inicialização de modelos que utilizam alguma forma de aleatoriedade internamente.

# Quando você define random_state para um número específico, como 0 ou 42, isso garante que os resultados sejam reproduzíveis. Ou seja, se você executar 
# o mesmo código várias vezes com o mesmo valor de random_state, você obterá os mesmos resultados. Isso é útil para garantir consistência em experimentos 
# ou em situações em que a aleatoriedade pode influenciar os resultados.

# No código, random_state=42 (por exemplo) foi usado para garantir que a divisão dos dados em conjuntos de treinamento e teste seja sempre a mesma, 
# independentemente de quantas vezes o código seja executado. O valor 42 é apenas um valor arbitrário escolhido por mim.

# Da mesma forma, random_state=0 e random_state=15 foram usados ao criar o classificador de árvore de decisão (DecisionTreeClassifier). Isso garante que a 
# inicialização interna do classificador também seja consistente, tornando os resultados reproduzíveis.

# Escolher 0 e 42 não é necessariamente importante em si, mas é comum escolher números que sejam fáceis de lembrar ou que sejam significativos para o 
# contexto do problema. O importante é usar o mesmo valor de random_state sempre que você precisar de resultados reproduzíveis.
