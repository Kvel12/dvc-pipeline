#!/usr/bin/env python3
"""
Genera predicciones para el conjunto de test
"""
import numpy as np
import pandas as pd
import pickle
import os

def main():
    print("=" * 60)
    print("ETAPA 4: PREDICCIÓN EN TEST SET")
    print("=" * 60)
    
    # Cargar modelo y scaler
    print("\nCargando modelo y scaler...")
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("Modelo y scaler cargados")
    
    # Cargar datos de test
    print("\nCargando datos de test...")
    test_df = pd.read_csv('data/test.csv')
    print(f"  - Shape: {test_df.shape}")
    
    # Preparar features
    features = ['time', 'temperature', 'cpu_usage', 'gpu_usage']
    X_test = test_df[features].values
    test_ids = test_df['ID'].values
    
    print(f"\nFeatures: {features}")
    print(f"  - X_test shape: {X_test.shape}")
    
    # Escalar y predecir
    print("\nGenerando predicciones...")
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    print(f" Predicciones generadas")
    print(f"\nDistribución de predicciones:")
    print(f"  - No Fallos (0): {np.sum(predictions == 0)} ({np.mean(predictions == 0)*100:.1f}%)")
    print(f"  - Fallos (1):    {np.sum(predictions == 1)} ({np.mean(predictions == 1)*100:.1f}%)")
    
    # Crear submission
    print("\nCreando archivo de submission...")
    os.makedirs('results', exist_ok=True)
    
    submission_df = pd.DataFrame({
        'ID': test_ids.astype(int),
        'Fallos': predictions.astype(int)
    })
    
    submission_df.to_csv('results/submission-v2.csv', index=False)
    
    print("results/submission-v2.csv")
    print(f"\nPrimeras 10 predicciones:")
    print(submission_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("PREDICCIÓN COMPLETADA")
    print("=" * 60)
    print(f"\nArchivo listo para submission: results/submission-v2.csv")

if __name__ == "__main__":
    main()