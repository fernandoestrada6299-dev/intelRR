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
    dim = TOTAL_WEIGHTS

    results_pso = []
    results_ga = []
    results_hybrid = []

    curves_pso = []
    curves_ga = []
    curves_hybrid = []

    test_acc_best_pso = []
    test_acc_best_ga = []
    test_acc_best_hybrid = []

    for _ in range(n_runs):
        pso = PSO(
            fitness_function=nn_fitness,
            dim=dim,
            n_particles=30,
            max_iter=max_iter,
            bounds=bounds,
        )
        best_pso, fit_pso, curve_pso = pso.optimize()

        ga = GA(
            fitness_function=nn_fitness,
            dim=dim,
            pop_size=40,
            max_iter=max_iter,
            bounds=bounds,
        )
        best_ga, fit_ga, curve_ga = ga.optimize()

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

        # Evaluar en test set la mejor solución de cada algoritmo
        test_acc_best_pso.append(evaluate_on_test(best_pso))
        test_acc_best_ga.append(evaluate_on_test(best_ga))
        test_acc_best_hybrid.append(evaluate_on_test(best_h))

    # Guardar CSV
    df = pd.DataFrame({
        "PSO_fitness": results_pso,
        "GA_fitness": results_ga,
        "Hybrid_fitness": results_hybrid,
        "PSO_test_acc": test_acc_best_pso,
        "GA_test_acc": test_acc_best_ga,
        "Hybrid_test_acc": test_acc_best_hybrid,
    })
    df.to_csv("results_nn.csv", index=False)

    def stats(arr):
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    print("== Neural Network - Estadísticos (fitness) ==")
    print("PSO:", stats(results_pso))
    print("GA:", stats(results_ga))
    print("Hybrid:", stats(results_hybrid))

    print("== Neural Network - Estadísticos (accuracy test) ==")
    print("PSO test acc:", stats(test_acc_best_pso))
    print("GA test acc:", stats(test_acc_best_ga))
    print("Hybrid test acc:", stats(test_acc_best_hybrid))

    # Gráficas
    plot_convergence(
        [curves_pso[0], curves_ga[0], curves_hybrid[0]],
        ["PSO", "GA", "Hybrid PSO-GA"],
        title="Curvas de convergencia - Red Neuronal",
    )

    plot_boxplot(
        [results_pso, results_ga, results_hybrid],
        ["PSO", "GA", "Hybrid"],
        title="Boxplot - NN (fitness)",
    )

    plot_histograms(
        [results_pso, results_ga, results_hybrid],
        ["PSO", "GA", "Hybrid"],
        title="Histogramas - NN (fitness)",
    )