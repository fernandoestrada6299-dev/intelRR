import numpy as np


class GA:
    """
    Algoritmo Genético real-valued para minimización de una función objetivo.
    """

    def __init__(
        self,
        fitness_function,
        dim,
        pop_size=40,
        max_iter=100,
        mutation_rate=0.1,
        bounds=(-1.0, 1.0),
    ):
        self.fitness_function = fitness_function
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.bounds = bounds

        low, high = bounds
        self.population = np.random.uniform(low, high, (pop_size, dim))
        self.convergence_curve = []

    def _crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def _mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.normal(0, 0.1, size=self.dim)
            individual = individual + mutation
        low, high = self.bounds
        return np.clip(individual, low, high)

    def optimize(self):
        for _ in range(self.max_iter):
            fitness_vals = np.array(
                [self.fitness_function(ind) for ind in self.population]
            )
            self.convergence_curve.append(np.min(fitness_vals))

            # Selección (elite simple: mejor mitad)
            elite_idx = np.argsort(fitness_vals)[: self.pop_size // 2]
            parents = self.population[elite_idx]

            # Nueva población
            new_pop = []
            for _ in range(self.pop_size):
                p1 = parents[np.random.randint(len(parents))]
                p2 = parents[np.random.randint(len(parents))]
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)

            self.population = np.array(new_pop)

        final_fitness = np.array(
            [self.fitness_function(ind) for ind in self.population]
        )
        best_idx = np.argmin(final_fitness)
        best = self.population[best_idx]
        best_fit = final_fitness[best_idx]
        return best, best_fit, self.convergence_curve