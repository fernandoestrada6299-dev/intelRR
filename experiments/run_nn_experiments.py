import numpy as np


class PSO:
    """
    Implementación básica de Particle Swarm Optimization (PSO)
    para minimización de una función objetivo.
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
        for _ in range(self.max_iter):
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            cognitive = self.c1 * r1 * (self.pbest - self.positions)
            social = self.c2 * r2 * (self.gbest - self.positions)

            self.velocities = self.w * self.velocities + cognitive + social
            self.positions = self.positions + self.velocities

            # Respetar límites
            low, high = self.bounds
            self.positions = np.clip(self.positions, low, high)

            fitness_vals = np.array(
                [self.fitness_function(p) for p in self.positions]
            )

            # Actualizar pbest
            improved = fitness_vals < self.pbest_fitness
            self.pbest[improved] = self.positions[improved]
            self.pbest_fitness[improved] = fitness_vals[improved]

            # Actualizar gbest
            best_idx = np.argmin(fitness_vals)
            if fitness_vals[best_idx] < self.gbest_fitness:
                self.gbest_fitness = fitness_vals[best_idx]
                self.gbest = self.positions[best_idx].copy()

            self.convergence_curve.append(self.gbest_fitness)

        return self.gbest, self.gbest_fitness, self.convergence_curve