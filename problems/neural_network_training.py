import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar dataset Iris
X, y = load_iris(return_X_y=True)
X = (X - X.mean(axis=0)) / X.std(axis=0)  # normalización simple

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
    Convierte el vector de pesos en matrices W1, b1, W2, b2.
    """
    w = np.array(weight_vector, dtype=float)
    assert w.size == TOTAL_WEIGHTS

    start = 0

    # W1
    size = INPUT_DIM * HIDDEN_UNITS
    W1 = w[start:start + size].reshape(INPUT_DIM, HIDDEN_UNITS)
    start += size

    # b1
    size = HIDDEN_UNITS
    b1 = w[start:start + size]
    start += size

    # W2
    size = HIDDEN_UNITS * OUTPUT_DIM
    W2 = w[start:start + size].reshape(HIDDEN_UNITS, OUTPUT_DIM)
    start += size

    # b2
    size = OUTPUT_DIM
    b2 = w[start:start + size]

    return W1, b1, W2, b2


def _forward(X, weight_vector):
    """
    Propagación hacia adelante de la red MLP con ReLU + softmax.
    """
    W1, b1, W2, b2 = _decode_weights(weight_vector)

    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU

    logits = a1 @ W2 + b2

    # Softmax estable
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(logits_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def _accuracy(X, y, weight_vector):
    probs = _forward(X, weight_vector)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)


def nn_fitness(weight_vector):
    """
    Fitness = 1 - accuracy en el conjunto de entrenamiento.
    """
    acc = _accuracy(X_train, y_train, weight_vector)
    return 1.0 - acc


def evaluate_on_test(weight_vector):
    """
    Evalúa en test set la mejor solución.
    """
    return _accuracy(X_test, y_test, weight_vector)