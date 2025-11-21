"""
Módulo para ejecutar experimentos del entrenamiento de redes neuronales usando PSO, GA y un enfoque híbrido PSO–GA. Realiza múltiples corridas, calcula métricas, guarda resultados y genera gráficas comparativas.
"""

import numpy as np
import pandas as pd

from algorithms.pso import PSO
from algorithms.ga import GA
from algorithms.hybrid_pso_ga import HybridPSO_GA
from problems.neural_network_training import nn_fitness, TOTAL_WEIGHTS, evaluate_on_test
from utils.plotting import plot_convergence, plot_boxplot, plot_histograms


def run_nn_experiments(
    n_runs=20,
    bounds=(-1.0, 1.0),
    max_iter=80
):
    """
    Ejecuta múltiples experimentos de optimización del entrenamiento de una red neuronal mediante PSO, GA y Hybrid PSO–GA.

    Parámetros:
    - n_runs: número de corridas independientes.
    - bounds: límites mínimos y máximos para los pesos de la red neuronal.
    - max_iter: número máximo de iteraciones por algoritmo.

    Proceso:
    1. Ejecuta los tres algoritmos n_runs veces.
    2. Registra el mejor fitness, curvas de convergencia y accuracies en test.
    3. Guarda resultados en CSV.
    4. Imprime estadísticos en consola.
    5. Genera gráficas de convergencia, boxplot e histogramas.

    Retorna:
    Nada. La función se utiliza con fines experimentales.
    """
    # Dimensión del vector de pesos de la red neuronal
    dim = TOTAL_WEIGHTS

    # Listas para almacenar los resultados de fitness y curvas de convergencia
    results_pso = []
    results_ga = []
    results_hybrid = []

    curves_pso = []
    curves_ga = []
    curves_hybrid = []

    test_acc_best_pso = []
    test_acc_best_ga = []
    test_acc_best_hybrid = []

    # Bucle principal: ejecutar múltiples corridas independientes
    for _ in range(n_runs):
        # Ejecutar PSO
        pso = PSO(
            fitness_function=nn_fitness,
            dim=dim,
            n_particles=30,
            max_iter=max_iter,
            bounds=bounds,
        )
        best_pso, fit_pso, curve_pso = pso.optimize()

        # Ejecutar GA
        ga = GA(
            fitness_function=nn_fitness,
            dim=dim,
            pop_size=40,
            max_iter=max_iter,
            bounds=bounds,
        )
        best_ga, fit_ga, curve_ga = ga.optimize()

        # Ejecutar algoritmo híbrido PSO–GA
        hybrid = HybridPSO_GA(
            fitness_function=nn_fitness,
            dim=dim,
            bounds=bounds,
            pso_iter=max_iter // 2,
            ga_iter=max_iter // 2,
        )
        best_h, fit_h, curve_h = hybrid.optimize()

        results_pso.append(fit_pso)
        results_ga.append(fit_ga)
        results_hybrid.append(fit_h)

        curves_pso.append(curve_pso)
        curves_ga.append(curve_ga)
        curves_hybrid.append(curve_h)

        # Evaluar la mejor solución encontrada en el conjunto de prueba
        test_acc_best_pso.append(evaluate_on_test(best_pso))
        test_acc_best_ga.append(evaluate_on_test(best_ga))
        test_acc_best_hybrid.append(evaluate_on_test(best_h))

    # Guardar resultados en archivo CSV
    df = pd.DataFrame({
        "PSO_fitness": results_pso,
        "GA_fitness": results_ga,
        "Hybrid_fitness": results_hybrid,
        "PSO_test_acc": test_acc_best_pso,
        "GA_test_acc": test_acc_best_ga,
        "Hybrid_test_acc": test_acc_best_hybrid,
    })
    df.to_csv("results_nn.csv", index=False)

    # Función auxiliar para calcular estadísticos
    def stats(arr):
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    # Imprimir estadísticas de fitness y accuracy
    print("== Neural Network - Estadísticos (fitness) ==")
    print("PSO:", stats(results_pso))
    print("GA:", stats(results_ga))
    print("Hybrid:", stats(results_hybrid))

    print("== Neural Network - Estadísticos (accuracy test) ==")
    print("PSO test acc:", stats(test_acc_best_pso))
    print("GA test acc:", stats(test_acc_best_ga))
    print("Hybrid test acc:", stats(test_acc_best_hybrid))

    # Gráfica de convergencia
    plot_convergence(
        [curves_pso[0], curves_ga[0], curves_hybrid[0]],
        ["PSO", "GA", "Hybrid PSO-GA"],
        title="Curvas de convergencia - Red Neuronal",
    )

    # Gráfica boxplot
    plot_boxplot(
        [results_pso, results_ga, results_hybrid],
        ["PSO", "GA", "Hybrid"],
        title="Boxplot - NN (fitness)",
    )

    # Gráfica de histogramas
    plot_histograms(
        [results_pso, results_ga, results_hybrid],
        ["PSO", "GA", "Hybrid"],
        title="Histogramas - NN (fitness)",
    )