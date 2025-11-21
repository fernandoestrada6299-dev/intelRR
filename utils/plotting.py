"""
Módulo que contiene funciones de graficación para los experimentos de IntelRR. Incluye histogramas con ajuste automático de rangos y visualización comparativa entre algoritmos metaheurísticos.
"""

def plot_histograms(results, labels, title="Distribución de resultados"):
    """
    Genera histogramas comparativos para los resultados de fitness obtenidos por distintos algoritmos.

    Parámetros:
    - results: lista de listas con valores numéricos de fitness para cada algoritmo.
    - labels: etiquetas correspondientes a cada conjunto de resultados.
    - title: título de la gráfica.

    Características:
    - Ajuste automático del rango del eje X según valores mínimos y máximos globales.
    - Normalización opcional mediante `density=True`.
    - Histogramas con contornos y transparencia para comparación visual.

    Retorna:
    Nada. Muestra la gráfica en pantalla.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Obtener los valores mínimos y máximos reales entre todos los resultados
    min_global = min([min(r) for r in results])
    max_global = max([max(r) for r in results])

    # Agregar un margen para evitar que los datos queden pegados en los límites
    padding = (max_global - min_global) * 0.3
    if padding == 0:
        padding = 0.0001  # en caso de valores casi iguales

    x_min = min_global - padding
    x_max = max_global + padding

    # Crear figura para la visualización
    plt.figure(figsize=(12, 6))

    # Dibujar cada histograma con su respectiva etiqueta
    for r, label in zip(results, labels):
        plt.hist(r, bins=15, alpha=0.6, label=label,
                 density=True, edgecolor='black')

    # Ajustar los límites visibles del eje X según el rango calculado
    plt.xlim(x_min, x_max)

    plt.xlabel("Fitness")
    plt.ylabel("Densidad")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()