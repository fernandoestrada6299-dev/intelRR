"""
Módulo para ejecutar experimentos del problema de asignación de recursos en la nube (Cloud Allocation) utilizando los algoritmos PSO, GA y un enfoque híbrido PSO–GA. Este script ejecuta múltiples corridas, recolecta métricas, guarda resultados en CSV y genera gráficas comparativas.
"""

import numpy as np
import pandas as pd

from algorithms.pso import PSO
from algorithms.ga import GA
from algorithms.hybrid_pso_ga import HybridPSO_GA
from problems.cloud_allocation import cloud_fitness
from utils.plotting import plot_convergence, plot_boxplot, plot_histograms


def run_cloud_experiments(
    n_runs=30,
    dim=3,
    bounds=(0.01, 1.0),
    max_iter=100
):
    """
    Ejecuta experimentos repetidos del problema Cloud Allocation.

    Parámetros:
    - n_runs: número de corridas independientes.
    - dim: número de dimensiones del problema.
    - bounds: límites mínimos y máximos para cada variable.
    - max_iter: número máximo de iteraciones por algoritmo.

    Proceso:
    1. Ejecuta PSO, GA y Hybrid PSO–GA n_runs veces.
    2. Almacena el mejor fitness de cada ejecución.
    3. Guarda un archivo CSV con todos los resultados.
    4. Imprime estadísticos básicos en consola.
    5. Genera gráficas: curvas de convergencia, boxplot e histogramas.

    Retorna:
    Nada. La función se utiliza con fines experimentales.
    """
    # Listas para almacenar los resultados de fitness y curvas de convergencia
    results_pso = []
    results_ga = []
    results_hybrid = []

    curves_pso = []
    curves_ga = []
    curves_hybrid = []

    # Bucle principal: repetimos las corridas independientes
    for _ in range(n_runs):
        # Ejecutar PSO
        pso = PSO(
            fitness_function=cloud_fitness,
            dim=dim,
            n_particles=30,
            max_iter=max_iter,
            bounds=bounds,
        )
        _, fit_pso, curve_pso = pso.optimize()

        # Ejecutar GA
        ga = GA(
            fitness_function=cloud_fitness,
            dim=dim,
            pop_size=40,
            max_iter=max_iter,
            bounds=bounds,
        )
        _, fit_ga, curve_ga = ga.optimize()

        # Ejecutar algoritmo híbrido PSO–GA
        hybrid = HybridPSO_GA(
            fitness_function=cloud_fitness,
            dim=dim,
            bounds=bounds,
            pso_iter=max_iter // 2,
            ga_iter=max_iter // 2,
        )
        _, fit_h, curve_h = hybrid.optimize()

        # Guardar fitness y curva de esta corrida
        results_pso.append(fit_pso)
        results_ga.append(fit_ga)
        results_hybrid.append(fit_h)

        curves_pso.append(curve_pso)
        curves_ga.append(curve_ga)
        curves_hybrid.append(curve_h)

    # Guardar resultados en archivo CSV
    df = pd.DataFrame({
        "PSO": results_pso,
        "GA": results_ga,
        "Hybrid_PSO_GA": results_hybrid,
    })
    df.to_csv("results_cloud.csv", index=False)

    # Función auxiliar para calcular estadísticos
    def stats(arr):
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    # Estadísticos básicos en consola
    print("== Cloud Allocation - Estadísticos ==")
    print("PSO:", stats(results_pso))
    print("GA:", stats(results_ga))
    print("Hybrid:", stats(results_hybrid))

    # Gráfica de convergencia
    plot_convergence(
        [curves_pso[0], curves_ga[0], curves_hybrid[0]],
        ["PSO", "GA", "Hybrid PSO-GA"],
        title="Curvas de convergencia - Cloud Allocation",
    )

    # Gráfica boxplot
    plot_boxplot(
        [results_pso, results_ga, results_hybrid],
        ["PSO", "GA", "Hybrid"],
        title="Boxplot - Cloud Allocation",
    )

    # Gráfica de histogramas
    plot_histograms(
        [results_pso, results_ga, results_hybrid],
        ["PSO", "GA", "Hybrid"],
        title="Histogramas - Cloud Allocation",
    )