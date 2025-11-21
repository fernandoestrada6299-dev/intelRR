"""
Módulo que implementa un algoritmo híbrido PSO–GA. Combina la exploración global de Particle Swarm Optimization con la capacidad de refinamiento local de un Algoritmo Genético para mejorar la calidad de la solución final.
"""

from algorithms.pso import PSO
from algorithms.ga import GA


class HybridPSO_GA:
    """
    Clase HybridPSO_GA:
    Implementa una estrategia híbrida de optimización en dos fases. Primero ejecuta PSO para localizar una región prometedora del espacio de búsqueda; después, utiliza esa solución como semilla inicial en un Algoritmo Genético para mejorar o refinar la solución final. Este enfoque busca equilibrar exploración y explotación.
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
        """
        Inicializa el algoritmo híbrido PSO–GA.

        Parámetros:
        - fitness_function: función objetivo a minimizar.
        - dim: dimensionalidad del vector solución.
        - bounds: tupla (min, max) que define los límites permitidos para cada variable.
        - pso_iter: número de iteraciones a ejecutar con PSO.
        - ga_iter: número de iteraciones a ejecutar con GA.
        - n_particles: número de partículas para el PSO.
        - pop_size: tamaño de población para el GA.
        """
        self.fitness_function = fitness_function
        self.dim = dim
        self.bounds = bounds
        self.pso_iter = pso_iter
        self.ga_iter = ga_iter
        self.n_particles = n_particles
        self.pop_size = pop_size

    def optimize(self):
        """
        Ejecuta el proceso híbrido completo:

        1. Ejecuta PSO para encontrar una solución inicial de alta calidad.
        2. Inserta la mejor solución de PSO como individuo inicial en GA.
        3. Ejecuta GA para refinar la solución obtenida.
        4. Retorna el mejor individuo final, su fitness y la curva de convergencia combinada.

        Retorna:
        - best_ga: mejor individuo encontrado por GA.
        - fit_ga: valor de fitness correspondiente.
        - hybrid_curve: curva de convergencia combinada PSO + GA.
        """
        # Antes de ejecutar PSO: inicializar y ejecutar optimización PSO
        pso = PSO(
            fitness_function=self.fitness_function,
            dim=self.dim,
            n_particles=self.n_particles,
            max_iter=self.pso_iter,
            bounds=self.bounds,
        )
        best_pso, fit_pso, curve_pso = pso.optimize()

        # Antes de inicializar GA: crear instancia de GA
        ga = GA(
            fitness_function=self.fitness_function,
            dim=self.dim,
            pop_size=self.pop_size,
            max_iter=self.ga_iter,
            bounds=self.bounds,
        )
        # Antes de sobrescribir el individuo 0: insertar mejor solución PSO en población GA
        ga.population[0] = best_pso

        # Antes de ejecutar GA: optimización GA
        best_ga, fit_ga, curve_ga = ga.optimize()

        # Antes de unir curvas de convergencia: concatenar curvas PSO y GA para análisis
        hybrid_curve = list(curve_pso) + list(curve_ga)

        # Antes de retornar resultados finales: devolver solución refinada y curva
        return best_ga, fit_ga, hybrid_curve