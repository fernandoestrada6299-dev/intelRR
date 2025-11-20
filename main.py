from experiments.run_cloud_experiments import run_cloud_experiments
from experiments.run_nn_experiments import run_nn_experiments


if __name__ == "__main__":
    print("Ejecutando experimentos de Cloud Allocation...")
    run_cloud_experiments()

    print("\nEjecutando experimentos de Entrenamiento de Red Neuronal...")
    run_nn_experiments()