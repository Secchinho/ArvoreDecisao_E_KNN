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

    autovaloresOrdenados = autovalores[::-1]
    autovetoresOrdenados = autovetores[:, ::-1]
    autovetoresNovos = autovetoresOrdenados[:, :novaDimensao]
    projecaoNovoEixo = dadosCentralizados @ autovetoresNovos
    varianciaExplicada = autovaloresOrdenados / autovaloresOrdenados.sum()

    return projecaoNovoEixo, autovetoresNovos, mediaColunas, varianciaExplicada

def KNN(xTreino, yTreino, xNovo, k):
    distancias = []
    for i in range(len(xTreino)):
        d = np.sqrt(np.sum(pow(abs(xTreino[i] - xNovo), 2))) #Distância Euclidiana
        #d = sum(xTreino[i] - xNovo) #Distância de Manhattan
        distancias.append(d)

    idsOrdenados = np.argsort(distancias)[:k]
    return Counter(yTreino[idsOrdenados]).most_common(1)[0][0]

if __name__ == "__main__":
    xNovo = np.array([5.8, 2.7, 5.1, 1.9]) #novo ponto para classificação
    k = 3
    print("KNN:", KNN(x, y, xNovo, k))

    projecaoNovoEixo, autovetoresNovos, media, var = PCA(x, 3)
    print(f"Variância explicada: {var * 100}")

    zNovo = ((xNovo - media) / x.std(axis=0, ddof=1)) @ autovetoresNovos
    print("KNN após o PCA: ", KNN(projecaoNovoEixo, y, zNovo, k))
