import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(curves, labels, title="Curvas de convergencia"):
    plt.figure(figsize=(10, 6))
    for curve, label in zip(curves, labels):
        plt.plot(curve, label=label)
    plt.xlabel("Iteración")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplot(results, labels, title="Comparación estadística"):
    plt.figure(figsize=(8, 6))
    plt.boxplot(results, labels=labels, showmeans=True)
    plt.ylabel("Fitness")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def plot_histograms(results, labels, title="Distribución de resultados"):
    plt.figure(figsize=(10, 6))
    for r, label in zip(results, labels):
        plt.hist(r, bins=10, alpha=0.5, label=label, density=True)
    plt.xlabel("Fitness")
    plt.ylabel("Densidad")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()