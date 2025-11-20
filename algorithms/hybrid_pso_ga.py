from algorithms.pso import PSO
from algorithms.ga import GA


class HybridPSO_GA:
    """
    Híbrido simple PSO–GA:
    1) Ejecuta PSO para acercarse a una buena región.
    2) Usa la mejor solución de PSO como semilla en GA para refinar.
    """

    def __init__(
        self,
        fitness_function,
        dim,
        bounds=(-1.0, 1.0),
        pso_iter=50,
        ga_iter=50,
        n_particles=30,
        pop_size=40,
    ):
        self.fitness_function = fitness_function
        self.dim = dim
        self.bounds = bounds
        self.pso_iter = pso_iter
        self.ga_iter = ga_iter
        self.n_particles = n_particles
        self.pop_size = pop_size

    def optimize(self):
        # Paso 1: PSO
        pso = PSO(
            fitness_function=self.fitness_function,
            dim=self.dim,
            n_particles=self.n_particles,
            max_iter=self.pso_iter,
            bounds=self.bounds,
        )
        best_pso, fit_pso, curve_pso = pso.optimize()

        # Paso 2: GA inicializado con mejor de PSO
        ga = GA(
            fitness_function=self.fitness_function,
            dim=self.dim,
            pop_size=self.pop_size,
            max_iter=self.ga_iter,
            bounds=self.bounds,
        )
        # Sobrescribir un individuo con la mejor solución de PSO
        ga.population[0] = best_pso

        best_ga, fit_ga, curve_ga = ga.optimize()

        # Curva híbrida = concatenación (opcional para gráficas)
        hybrid_curve = list(curve_pso) + list(curve_ga)

        # Tomamos como resultado final el de GA
        return best_ga, fit_ga, hybrid_curve