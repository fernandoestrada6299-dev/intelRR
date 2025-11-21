"""
Módulo que implementa el algoritmo Particle Swarm Optimization (PSO) para la minimización de funciones objetivo continuas. El método utiliza un conjunto de partículas que se desplazan por el espacio de búsqueda guiadas por su experiencia individual y colectiva.
"""

import numpy as np


class PSO:
    """
    Clase PSO:
    Implementa el algoritmo Particle Swarm Optimization para optimización continua en problemas de minimización. Incluye componentes de inercia, atracción cognitiva y atracción social, además de un manejo automático de límites mediante clipping.
    """

    def __init__(
        self,
        fitness_function,
        dim,
        n_particles=30,
        max_iter=100,
        w=0.7,
        c1=1.5,
        c2=1.5,
        bounds=(-1.0, 1.0),
    ):
        """
        Inicializa los parámetros del algoritmo PSO.

        Parámetros:
        - fitness_function: función objetivo a minimizar.
        - dim: número de dimensiones del vector solución.
        - n_particles: cantidad de partículas en el enjambre.
        - max_iter: número máximo de iteraciones.
        - w: coeficiente de inercia del movimiento.
        - c1: coeficiente cognitivo (influencia del mejor personal).
        - c2: coeficiente social (influencia del mejor global).
        - bounds: tupla (min, max) que define los límites permitidos para cada variable.
        """
        self.fitness_function = fitness_function
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds

        low, high = bounds
        self.positions = np.random.uniform(low, high, (n_particles, dim))
        self.velocities = np.zeros((n_particles, dim))

        self.pbest = self.positions.copy()
        self.pbest_fitness = np.array(
            [self.fitness_function(p) for p in self.positions]
        )

        best_idx = np.argmin(self.pbest_fitness)
        self.gbest = self.positions[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]

        self.convergence_curve = []

    def optimize(self):
        """
        Ejecuta el ciclo completo de optimización PSO.

        En cada iteración:
        1. Actualiza las velocidades usando los términos de inercia, cognitivo y social.
        2. Actualiza posiciones de las partículas.
        3. Aplica clipping para respetar los límites del dominio.
        4. Evalúa la función objetivo.
        5. Actualiza los mejores valores personales y globales.
        6. Almacena la curva de convergencia.

        Retorna:
        - gbest: mejor solución encontrada.
        - gbest_fitness: mejor fitness obtenido.
        - convergence_curve: lista con la evolución del mejor fitness por iteración.
        """
        for _ in range(self.max_iter):
            # Coeficientes aleatorios para los términos cognitivo y social
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            # Cálculo de los términos cognitivo y social que guían el movimiento de las partículas
            cognitive = self.c1 * r1 * (self.pbest - self.positions)
            social = self.c2 * r2 * (self.gbest - self.positions)

            # Ecuación principal de actualización de velocidades: inercia + atracción cognitiva + atracción social
            self.velocities = self.w * self.velocities + cognitive + social
            self.positions = self.positions + self.velocities

            # Respetar límites para evitar que las partículas salgan del dominio permitido
            low, high = self.bounds
            self.positions = np.clip(self.positions, low, high)

            fitness_vals = np.array(
                [self.fitness_function(p) for p in self.positions]
            )

            # Actualizar mejores personales (pbest) y global (gbest) según los valores de fitness obtenidos
            improved = fitness_vals < self.pbest_fitness
            self.pbest[improved] = self.positions[improved]
            self.pbest_fitness[improved] = fitness_vals[improved]

            best_idx = np.argmin(fitness_vals)
            if fitness_vals[best_idx] < self.gbest_fitness:
                self.gbest_fitness = fitness_vals[best_idx]
                self.gbest = self.positions[best_idx].copy()

            self.convergence_curve.append(self.gbest_fitness)

        return self.gbest, self.gbest_fitness, self.convergence_curve