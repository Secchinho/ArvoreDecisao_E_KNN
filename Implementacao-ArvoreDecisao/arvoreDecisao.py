import math
from collections import Counter

exemplos = [
    {"Tempo": "sol",     "Temperatura": "quente", "Umidade": "alta",   "Vento": "não", "Jogar": "não"},
    {"Tempo": "sol",     "Temperatura": "quente", "Umidade": "alta",   "Vento": "sim", "Jogar": "não"},
    {"Tempo": "nublado", "Temperatura": "quente", "Umidade": "alta",   "Vento": "não", "Jogar": "sim"},
    {"Tempo": "chuva",   "Temperatura": "médio",  "Umidade": "alta",   "Vento": "não", "Jogar": "sim"},
    {"Tempo": "chuva",   "Temperatura": "frio",   "Umidade": "normal", "Vento": "não", "Jogar": "sim"},
    {"Tempo": "chuva",   "Temperatura": "frio",   "Umidade": "normal", "Vento": "sim", "Jogar": "não"},
    {"Tempo": "nublado", "Temperatura": "frio",   "Umidade": "normal", "Vento": "sim", "Jogar": "sim"},
    {"Tempo": "sol",     "Temperatura": "médio",  "Umidade": "alta",   "Vento": "não", "Jogar": "não"},
    {"Tempo": "sol",     "Temperatura": "frio",   "Umidade": "normal", "Vento": "não", "Jogar": "sim"},
    {"Tempo": "chuva",   "Temperatura": "médio",  "Umidade": "normal", "Vento": "não", "Jogar": "sim"},
    {"Tempo": "sol",     "Temperatura": "médio",  "Umidade": "normal", "Vento": "sim", "Jogar": "sim"},
    {"Tempo": "nublado", "Temperatura": "médio",  "Umidade": "alta",   "Vento": "sim", "Jogar": "sim"},
    {"Tempo": "nublado", "Temperatura": "quente", "Umidade": "normal", "Vento": "não", "Jogar": "sim"},
    {"Tempo": "chuva",   "Temperatura": "médio",  "Umidade": "alta",   "Vento": "sim", "Jogar": "não"},
]

atributos = ["Tempo", "Temperatura", "Umidade", "Vento"]

def entropia(exemplos):
    classes = [e["Jogar"] for e in exemplos]
    contagens = Counter(classes)
    total = len(exemplos)

    entr = 0
    for qtd in contagens.values():
        p = qtd / total
        entr -= p * math.log2(p)
    return entr

def valorDaMaioria(exemplos):
    classes = [e["Jogar"] for e in exemplos]
    moda = Counter(classes).most_common(1)[0][0]

    return moda

def ganhoDeInformacao(atributo, exemplos):
    soma = 0

    valores = set()
    for e in exemplos:
        valores.add(e[atributo])

    for v in valores:
        subconjunto = []
        for e in exemplos:
            if e[atributo] == v:
                subconjunto.append(e)

        peso = len(subconjunto) / len(exemplos)
        soma += peso * entropia(subconjunto)
    return entropia(exemplos) - soma

def aprendizagem(exemplos, atributos, exemplosPais):
    if exemplos == []:
        return valorDaMaioria(exemplosPais)
    elif entropia(exemplos) == 0:
        classes = [e["Jogar"] for e in exemplos]
        return classes[0]
    elif atributos == []:
        return valorDaMaioria(exemplos)
    else:
        A = max(atributos, key=lambda a: ganhoDeInformacao(a, exemplos))
        arvore = {A: {}}
        novosAtributos = []
        for a in atributos:
            if a != A:
                novosAtributos.append(a)

        for v in set(e[A] for e in exemplos):
            exs = []
            for e in exemplos:
                if e[A] == v:
                    exs.append(e)

            subarvore = aprendizagem(exs, novosAtributos, exemplos)
            arvore[A][v] = subarvore

        return arvore

if __name__ == "__main__":
    print(len(exemplos))
    print(entropia(exemplos))
    print(valorDaMaioria(exemplos))
    print(aprendizagem(exemplos, atributos, []))
