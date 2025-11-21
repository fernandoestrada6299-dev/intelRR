"""
Módulo principal del proyecto IntelRR. Coordina la ejecución de los experimentos de Cloud Allocation y Neural Network Training.
"""
from experiments.run_cloud_experiments import run_cloud_experiments
from experiments.run_nn_experiments import run_nn_experiments


if __name__ == "__main__":
    # Ejecutar experimentos del problema de asignación de recursos en la nube
    print("Ejecutando experimentos de Cloud Allocation...")
    run_cloud_experiments()

    # Ejecutar experimentos del problema de entrenamiento de red neuronal optimizada por metaheurísticas
    print("\nEjecutando experimentos de Entrenamiento de Red Neuronal...")
    run_nn_experiments()