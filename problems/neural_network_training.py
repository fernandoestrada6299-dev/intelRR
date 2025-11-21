"""
Módulo para el entrenamiento y evaluación de una red neuronal simple (MLP) optimizada mediante metaheurísticas. Incluye funciones para decodificar los pesos, realizar la propagación hacia adelante, calcular exactitud y definir la función de aptitud.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar dataset Iris
X, y = load_iris(return_X_y=True)
# Normalización simple: centrar y escalar características
X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalización simple

# División en conjuntos de entrenamiento y prueba con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

INPUT_DIM = X.shape[1]      # 4
HIDDEN_UNITS = 5
OUTPUT_DIM = len(np.unique(y))  # 3

# Total de parámetros de la red: W1 + b1 + W2 + b2
TOTAL_WEIGHTS = (
    INPUT_DIM * HIDDEN_UNITS
    + HIDDEN_UNITS
    + HIDDEN_UNITS * OUTPUT_DIM
    + OUTPUT_DIM
)


def _decode_weights(weight_vector):
    """
    Convierte el vector plano de pesos en las matrices y vectores correspondientes a la arquitectura MLP:
    - W1: pesos de entrada → capa oculta
    - b1: bias de capa oculta
    - W2: pesos de capa oculta → salida
    - b2: bias de salida
    """
    w = np.array(weight_vector, dtype=float)
    assert w.size == TOTAL_WEIGHTS

    start = 0

    # Extraer W1: pesos de entrada a capa oculta
    size = INPUT_DIM * HIDDEN_UNITS
    W1 = w[start:start + size].reshape(INPUT_DIM, HIDDEN_UNITS)
    start += size

    # Extraer b1: bias de capa oculta
    size = HIDDEN_UNITS
    b1 = w[start:start + size]
    start += size

    # Extraer W2: pesos de capa oculta a salida
    size = HIDDEN_UNITS * OUTPUT_DIM
    W2 = w[start:start + size].reshape(HIDDEN_UNITS, OUTPUT_DIM)
    start += size

    # Extraer b2: bias de capa salida
    size = OUTPUT_DIM
    b2 = w[start:start + size]

    return W1, b1, W2, b2


def _forward(X, weight_vector):
    """
    Realiza la propagación hacia adelante del MLP usando ReLU en la capa oculta
    y softmax en la capa de salida.

    Parámetros:
    - X: matriz de entrada
    - weight_vector: vector plano con todos los pesos y bias

    Retorna:
    - Matriz de probabilidades por clase (softmax)
    """
    W1, b1, W2, b2 = _decode_weights(weight_vector)

    z1 = X @ W1 + b1
    # Aplicar función ReLU en capa oculta
    a1 = np.maximum(0, z1)  # ReLU

    logits = a1 @ W2 + b2

    # Softmax estable para obtener probabilidades
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def _accuracy(X, y, weight_vector):
    """
    Calcula la exactitud (accuracy) comparando predicciones con etiquetas reales.
    """
    probs = _forward(X, weight_vector)
    # Obtener predicciones de clase como índice con mayor probabilidad
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)


def nn_fitness(weight_vector):
    """
    Función de aptitud usada por metaheurísticas.
    Devuelve: 1 - accuracy en el conjunto de entrenamiento.
    El objetivo es minimizar este valor.
    """
    # Calcular accuracy en conjunto de entrenamiento
    acc = _accuracy(X_train, y_train, weight_vector)
    return 1.0 - acc


def evaluate_on_test(weight_vector):
    """
    Evalúa la mejor solución encontrada en el conjunto de prueba (test).
    Retorna el valor de exactitud.
    """
    # Retornar accuracy en conjunto de prueba
    return _accuracy(X_test, y_test, weight_vector)