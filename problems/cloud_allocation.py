"""
Módulo que define la función de aptitud (fitness) para el problema de asignación de recursos en la nube. Evalúa el costo total combinando tiempo de respuesta y consumo energético.
"""
import numpy as np

def cloud_fitness(x):
    """
    Calcula el valor de fitness para una solución del problema de asignación de recursos en la nube.

    Parámetros:
    - x: vector de tres elementos [CPU, Memoria, Energía].

    Comportamiento:
    - x[0] representa la proporción de CPU asignada.
    - x[1] representa la proporción de memoria asignada.
    - x[2] representa el nivel relativo de consumo energético.
    - Se aseguran valores mínimos de CPU y memoria para evitar divisiones entre cero.
    - El tiempo de respuesta se calcula como: (1 / CPU) + (0.5 / Memoria).
    - El costo energético es: Energía².

    Retorna:
    - El valor total de la función objetivo: tiempo de respuesta + costo energético.
    """
    cpu, mem, energy = x

    # Evitar valores muy pequeños que causen inestabilidad
    cpu = max(cpu, 0.01)
    mem = max(mem, 0.01)

    # Cálculo del tiempo de respuesta del sistema
    time_response = (1.0 / cpu) + (0.5 / mem)
    # Penalización cuadrática por consumo energético
    energy_cost = energy ** 2

    return time_response + energy_cost