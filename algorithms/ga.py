"""
Módulo GA: Implementación de un Algoritmo Genético (Genetic Algorithm) para optimización continua.
Este algoritmo busca minimizar una función objetivo usando operadores de selección, cruce y mutación.
"""

import numpy as np


class GA:
    """
    Algoritmo Genético de valores reales (real-valued GA) para la minimización
    de funciones objetivo continuas. Incluye selección elitista, cruce
    intermedio y mutación gaussiana.
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
        """
        Inicializa el Algoritmo Genético.

        Parámetros:
        - fitness_function: función objetivo a minimizar.
        - dim: dimensión del vector solución.
        - pop_size: número de individuos en la población.
        - max_iter: número máximo de iteraciones.
        - mutation_rate: probabilidad de mutar un individuo.
        - bounds: tupla (min, max) que define los límites permitidos para cada gen.
        """
        self.fitness_function = fitness_function
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.bounds = bounds

        low, high = bounds
        # Inicializa la población con valores aleatorios uniformes dentro de los límites
        self.population = np.random.uniform(low, high, (pop_size, dim))
        self.convergence_curve = []

    def _crossover(self, parent1, parent2):
        """
        Realiza el cruce entre dos padres usando interpolación lineal (blend crossover).
        Retorna un nuevo individuo hijo.
        """
        alpha = np.random.rand()
        # Combina genes de ambos padres según un peso aleatorio alpha
        return alpha * parent1 + (1 - alpha) * parent2

    def _mutate(self, individual):
        """
        Aplica mutación gaussiana a un individuo según mutation_rate.
        Los valores mutados se mantienen dentro de los límites usando clipping.
        """
        if np.random.rand() < self.mutation_rate:
            # Genera una mutación gaussiana con media 0 y desviación estándar 0.1
            mutation = np.random.normal(0, 0.1, size=self.dim)
            individual = individual + mutation
        low, high = self.bounds
        # Asegura que los valores mutados estén dentro de los límites permitidos
        return np.clip(individual, low, high)

    def optimize(self):
        """
        Ejecuta el ciclo evolutivo completo del GA.

        - Evalúa la población.
        - Selecciona la élite (mejor mitad).
        - Genera una nueva población mediante cruce + mutación.
        - Registra la curva de convergencia.

        Retorna:
        - best: el mejor individuo encontrado.
        - best_fit: su valor de fitness.
        - convergence_curve: historial del mejor fitness por iteración.
        """
        for _ in range(self.max_iter):
            # Evaluar fitness de todos los individuos en la población actual
            fitness_vals = np.array(
                [self.fitness_function(ind) for ind in self.population]
            )
            # Registrar el mejor fitness de esta iteración para el seguimiento de la convergencia
            self.convergence_curve.append(np.min(fitness_vals))

            # Selección: elegir la élite que corresponde a la mejor mitad de la población
            elite_idx = np.argsort(fitness_vals)[: self.pop_size // 2]
            parents = self.population[elite_idx]

            # Generar nueva población mediante cruce y mutación de los padres seleccionados
            new_pop = []
            for _ in range(self.pop_size):
                p1 = parents[np.random.randint(len(parents))]
                p2 = parents[np.random.randint(len(parents))]
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_pop.append(child)

            self.population = np.array(new_pop)

        # Evaluar fitness final para determinar el mejor individuo encontrado
        final_fitness = np.array(
            [self.fitness_function(ind) for ind in self.population]
        )
        best_idx = np.argmin(final_fitness)
        best = self.population[best_idx]
        best_fit = final_fitness[best_idx]
        return best, best_fit, self.convergence_curve