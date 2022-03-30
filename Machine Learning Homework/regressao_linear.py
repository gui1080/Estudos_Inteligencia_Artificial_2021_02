# --------------------------
# Exemplos da Huaweii
# UnB - HCIA - Turma 1
# --------------------------
# Aluno Guilherme Braga, 17/0162290
# Estudo de Exercícios de Machine Learning, entrega 30/03/2022
# Usuário gui1080 no Github
# --------------------------
# Fontes (Gabriel Ferreira/Huaweii)
# Aula: https://www.youtube.com/watch?v=00YVrBhHucY
# Repositório: https://gitlab.com/Gabrielcarvfer/hcia-ai-ml-examples-unb/-/tree/master/ 
# --------------------------
# Testado no Ubuntu 20.04.4 LTS, com Python 3 e Visual Studio Code
# --------------------------

import matplotlib.pyplot as plt  # Biblioteca para plotagem
import random

pares_xy = [
    # (x_i, y_i),
    (3, 1),
    (21, 10),
    (22, 14),
    (34, 34),
    (54, 44),
    (34, 36),
    (55, 22),
    (67, 67),
    (89, 79),
    (99, 90)
]

# Separa os x e os y em listas diferentes
x = [par[0] for par in pares_xy]
y = [par[1] for par in pares_xy]

# mostra no gráfico
plt.scatter(x, y)
plt.xlabel("Valores X")
plt.ylabel("Valores Y")
plt.show()

# --------------------------

def funcao_linear(a, b, x):
    return a * x + b

# --------------------------

# residuo é a diferença entre o medido e o estimado ao quadrado
def residuo_metodo_minimos_quadrados(y_medido, y_aproximado):
    residuo = 0
    for i in range(len(y_medido)):
        residuo += (y_medido[i] - y_aproximado[i]) ** 2
    return residuo

# --------------------------

def gera_y_aproximado(a, b, x_medido):
    numeros = len(x_medido)
    y_aproximado = []
    # Para cada x_i, gera o valor aproximado de y_i(x_i) = a*x_i + b
    for i in range(numeros):
        y_aproximado.append(funcao_linear(a, b, x_medido[i]))
    return y_aproximado

# --------------------------

def iteracao(a, b, x_medido, y_medido, taxa_aprendizagem):
    numeros = len(y_medido)

    y_aproximado = gera_y_aproximado(a, b, x_medido)

    # Calcula o ajuste para parâmetros 'a' e 'b'
    residuo_a = sum([(y_aproximado[i] - y_medido[i]) * x_medido[i] for i in range(numeros)]) / numeros
    residuo_b = sum([(y_aproximado[i] - y_medido[i]) for i in range(numeros)]) / numeros

    # Retorna os valores de 'a' e 'b' ajustados
    a = a - taxa_aprendizagem * residuo_a
    b = b - taxa_aprendizagem * residuo_b

    return a, b

# --------------------------
# função principal
def minimizacao_residuo(x_medido, y_medido, taxa_aprendizagem, iteracoes=100):
    
    # Escolhemos 'a' e 'b' iniciais aleatoriamente
    a = random.random()
    b = random.random()

    # Depois executamos uma série de iterações,
    # ajustando os valores de 'a' e 'b' progressivamente
    for it in range(iteracoes):
        
        a, b = iteracao(a, b, x_medido, y_medido, taxa_aprendizagem) # novo "a" e "b" dado um resíduo, comparando o medido e o calculado 
                                                                     # com a função linear

        y_aproximado = gera_y_aproximado(a, b, x_medido) # pra cada x e y, gera y = a * x + b
        
        residuo = residuo_metodo_minimos_quadrados(y_medido, y_aproximado) # com a e b aleatório em primeira iteração, pega resíduo
        
        print("Iteração %d: a=%.3f, b=%.3f, residuo %.1f" % (it, a, b, residuo))

        if it % int(iteracoes / 5) == 0 or it == (iteracoes - 1):
            plt.plot(x_medido, y_aproximado, label="y_aprox it=%d residuo=%.1f" % (it, residuo))

    plt.scatter(x_medido, y_medido, label="y_medido")
    plt.legend()
    plt.show()

# --------------------------

# 1^-5 é a taxa de aprendizagem
minimizacao_residuo(x, y, 1e-5)

# 1^-6 é a taxa de aprendizagem
minimizacao_residuo(x, y, 1e-6, iteracoes=500)

# 1^⁻3 é a taxa de aprendizagem, com menos iterações
minimizacao_residuo(x, y, 1e-3, iteracoes=10)
