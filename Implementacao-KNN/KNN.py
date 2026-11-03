import numpy as np
from collections import Counter

x = np.array([
    [5.0, 3.4, 1.5, 0.4],
    [4.3, 3.0, 1.1, 0.1],
    [6.3, 2.8, 5.1, 1.5],
    [5.8, 2.7, 5.1, 1.9],
    [4.9, 2.4, 3.3, 1.0],
    [6.9, 3.1, 4.9, 1.5],
])

y = np.array([
    "Iris-setosa", "Iris-setosa", "Iris-virginica", "Iris-virginica",
    "Iris-versicolor", "Iris-versicolor",
])

def PCA(x, novaDimensao):
    mediaColunas = x.mean(axis=0)
    #print(mediaColunas, "\n")

    dadosCentralizados = (x - mediaColunas) / x.std(axis=0, ddof=1)
    #print(dadosCentralizados, "\n")

    matrizCovariancia = np.cov(dadosCentralizados, rowvar=False)
    #print(matrizCovariancia, "\n")

    autovalores, autovetores = np.linalg.eigh(matrizCovariancia)
    #print(autovalores, "\n\n", autovetores, "\n")

    autovaloresOrdenados = autovalores[::-1] #esse é o mecanismo de slice do python aqui eu estou indo até o fim
                                             #do vetor e voltando com o passo de -1

    autovetoresOrdenados = autovetores[:, ::-1] #mesma coisa, mas aqui vou voltando de coluna em coluna
                                                #pois os autovetores são as colunas

    #Eu posso ir de trás pra frente pois a função do numpy (np.linalg.eigh) já me devolve os autovalores e
    #autovetores na ordem crescente de módulo, mas no caso aqui eu quero decrescente, por isso basta inverter

    autovetoresNovos = autovetoresOrdenados[:, :novaDimensao] #Aqui eu pego apenas os primeiros N autovetores
                                                              #Esse N corresponde as dimensões desejadas

    projecaoNovoEixo = dadosCentralizados @ autovetoresNovos #o @ é pra multiplicação de matrizes

    varianciaExplicada = autovaloresOrdenados / autovaloresOrdenados.sum() #Aqui é apenas para ver quais
                                                                           #autovetores têm mais informação

    return projecaoNovoEixo, autovetoresNovos, mediaColunas, varianciaExplicada

def KNN(xTreino, yTreino, xNovo, k):
    distancias = []
    for i in range(len(xTreino)):
        d = np.sqrt(np.sum(pow(abs(xTreino[i] - xNovo), 2)))
        #d = sum(xTreino[i] - xNovo)
        distancias.append(d)

    idsOrdenados = np.argsort(distancias)[:k] #argsort informa a posição das distancias em ordem crescente
    #ou seja, se as distancias fossem [50, 20, 72, 14, 37], o argsort devolveria um vetor [3, 1, 4, 0, 2]

    #o Counter é a ferramenta sugerida pelo claude, pois ele já devolve o valor que mais ocorre em um vetor
    #Aqui estamos ordenando o yTreino (classes) de acordo com as distancias mais próximas e o Counter já devolve
    #quais aparecem mais frequentemente
    return Counter(yTreino[idsOrdenados]).most_common(1)[0][0]

if __name__ == "__main__":
    xNovo = np.array([5.8, 2.7, 5.1, 1.9]) #novo ponto para classificação
    k = 3
    print("KNN:", KNN(x, y, xNovo, k))

    projecaoNovoEixo, autovetoresNovos, media, var = PCA(x, 3)
    print(f"Variância explicada: {var * 100}")

    zNovo = ((xNovo - media) / x.std(axis=0, ddof=1)) @ autovetoresNovos # centralizando e projetando o novo ponto nas novas dimensões
    print("KNN após o PCA: ", KNN(projecaoNovoEixo, y, zNovo, k))
