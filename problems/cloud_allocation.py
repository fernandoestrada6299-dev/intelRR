import numpy as np

def cloud_fitness(x):
    """
    x[0] = proporción de CPU asignada
    x[1] = proporción de Memoria asignada
    x[2] = parámetro de Consumo energético relativo
    """
    cpu, mem, energy = x

    # Evitar valores peligrosos
    cpu = max(cpu, 0.01)
    mem = max(mem, 0.01)

    time_response = (1.0 / cpu) + (0.5 / mem)
    energy_cost = energy ** 2

    return time_response + energy_cost