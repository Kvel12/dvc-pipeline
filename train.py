#!/usr/bin/env python3
"""
Entrenamiento del modelo de detección de fallos
"""
import numpy as np
import pandas as pd
import pickle
import yaml
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_params():
    """Carga parámetros desde params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de clasificación"""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return {
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy)
    }

def main():
    print("=" * 60)
    print("ETAPA 3: ENTRENAMIENTO DEL MODELO")
    print("=" * 60)
    
    # Cargar parámetros
    params = load_params()
    print("\nParámetros del modelo:")
    print(f"  - C (regularización): {params['model']['C']}")
    print(f"  - Solver: {params['model']['solver']}")
    print(f"  - Max iterations: {params['model']['max_iter']}")
    print(f"  - Train size: {params['split']['train_size']*100}%")
    
    # Cargar datos procesados
    print("\nCargando datos procesados...")
    train_df = pd.read_csv('data/train_processed.csv')
    print(f"  - Shape: {train_df.shape}")
    
    # Preparar features y target
    features = ['time', 'temperature', 'cpu_usage', 'gpu_usage']
    X = train_df[features].values
    y = train_df['Fallos'].values
    
    print(f"\nFeatures utilizadas: {features}")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - Distribución target: No Fallo={np.sum(y==0)}, Fallo={np.sum(y==1)}")
    
    # División train/validation
    print("\nDividiendo datos...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        train_size=params['split']['train_size'],
        random_state=params['split']['random_state'],
        stratify=y
    )
    print(f"  - Train: {X_train.shape[0]} muestras")
    print(f"  - Validación: {X_val.shape[0]} muestras")
    
    # Escalado
    print("\nEscalando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Entrenamiento
    print("\nEntrenando modelo...")
    model = LogisticRegression(
        C=params['model']['C'],
        solver=params['model']['solver'],
        max_iter=params['model']['max_iter'],
        random_state=params['model']['random_state']
    )
    model.fit(X_train_scaled, y_train)
    
    print("Modelo entrenado")
    print(f"\nCoeficientes del modelo:")
    for feat, coef in zip(features, model.coef_[0]):
        print(f"  - {feat:15}: {coef:8.4f}")
    print(f"  - Intercepto:       {model.intercept_[0]:8.4f}")
    
    # Predicciones
    print("\nEvaluando en validación...")
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Métricas
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    print("\nMÉTRICAS DE ENTRENAMIENTO:")
    print(f"  - Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  - Precision: {train_metrics['precision']:.4f}")
    print(f"  - Recall:    {train_metrics['recall']:.4f}")
    print(f"  - F1-Score:  {train_metrics['f1_score']:.4f}")
    
    print("\nMÉTRICAS DE VALIDACIÓN:")
    print(f"  - Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  - Precision: {val_metrics['precision']:.4f}")
    print(f"  - Recall:    {val_metrics['recall']:.4f}")
    print(f"  - F1-Score:  {val_metrics['f1_score']:.4f}")
    
    print(f"\nMatriz de Confusión (Validación):")
    print(f"                 Predicción")
    print(f"             No Fallo  Fallo")
    print(f"Real No Fallo    {val_metrics['TN']:4d}   {val_metrics['FP']:4d}")
    print(f"Real Fallo       {val_metrics['FN']:4d}   {val_metrics['TP']:4d}")
    
    # Guardar modelo y scaler
    print("\nGuardando modelo y scaler...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("models/model.pkl")
    print("models/scaler.pkl")
    
    # Guardar métricas
    os.makedirs('metrics', exist_ok=True)
    metrics_output = {
        'train': train_metrics,
        'validation': val_metrics,
        'features': features
    }
    
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print("metrics/train_metrics.json")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"\nF1-Score Final: {val_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()