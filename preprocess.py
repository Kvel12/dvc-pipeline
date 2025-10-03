#!/usr/bin/env python3
"""
Preprocesamiento y análisis exploratorio de datos
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
import os

def load_params():
    """Carga parámetros desde params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_target_variable(df, params):
    """
    Crea la variable target basada en umbrales críticos
    
    Lógica: Al menos 2 condiciones críticas = FALLO
    """
    thresholds = params['thresholds']
    
    # Calcular umbral de performance
    perf_mean = df['performance'].mean()
    perf_threshold = perf_mean * thresholds['performance_drop']
    
    # Condiciones críticas individuales
    temp_critica = (df['temperature'] > thresholds['temperature']).astype(int)
    cpu_critica = (df['cpu_usage'] > thresholds['cpu_usage']).astype(int)
    gpu_critica = (df['gpu_usage'] > thresholds['gpu_usage']).astype(int)
    performance_critica = (df['performance'] < perf_threshold).astype(int)
    
    # Lógica combinada: Al menos 2 condiciones = FALLO
    condiciones_criticas = temp_critica + cpu_critica + gpu_critica + performance_critica
    target = (condiciones_criticas >= 2).astype(int)
    
    # Estadísticas
    stats = {
        'performance_mean': float(perf_mean),
        'performance_threshold': float(perf_threshold),
        'temp_critica_pct': float(temp_critica.mean() * 100),
        'cpu_critica_pct': float(cpu_critica.mean() * 100),
        'gpu_critica_pct': float(gpu_critica.mean() * 100),
        'performance_critica_pct': float(performance_critica.mean() * 100),
        'fallos_pct': float(target.mean() * 100)
    }
    
    return target, stats

def create_visualizations(df, target):
    """Crea visualizaciones de distribuciones"""
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(20, 5))
    
    # Temperatura
    plt.subplot(1, 5, 1)
    plt.hist(df['temperature'], bins=50, alpha=0.7, color='red')
    plt.axvline(x=90, color='red', linestyle='--', label='Crítico >90°C')
    plt.title('Distribución Temperatura')
    plt.xlabel('Temperatura')
    plt.legend()
    
    # CPU
    plt.subplot(1, 5, 2)
    plt.hist(df['cpu_usage'], bins=50, alpha=0.7, color='blue')
    plt.axvline(x=90, color='blue', linestyle='--', label='Crítico >90%')
    plt.title('Distribución CPU Usage')
    plt.xlabel('CPU Usage')
    plt.legend()
    
    # GPU
    plt.subplot(1, 5, 3)
    plt.hist(df['gpu_usage'], bins=50, alpha=0.7, color='green')
    plt.axvline(x=90, color='green', linestyle='--', label='Crítico >90%')
    plt.title('Distribución GPU Usage')
    plt.xlabel('GPU Usage')
    plt.legend()
    
    # Performance
    plt.subplot(1, 5, 4)
    plt.hist(df['performance'], bins=50, alpha=0.7, color='orange')
    perf_mean = df['performance'].mean()
    perf_threshold = perf_mean * 0.8
    plt.axvline(x=perf_threshold, color='orange', linestyle='--', 
                label=f'Crítico <{perf_threshold:.3f}')
    plt.title('Distribución Performance')
    plt.xlabel('Performance')
    plt.legend()
    
    # Time
    plt.subplot(1, 5, 5)
    plt.hist(df['time'], bins=50, alpha=0.7, color='purple')
    plt.title('Distribución Time')
    plt.xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('plots/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gráficas guardadas en plots/distributions.png")

def main():
    print("=" * 60)
    print("ETAPA 2: PREPROCESAMIENTO Y ANÁLISIS")
    print("=" * 60)
    
    # Cargar parámetros
    params = load_params()
    print("\nParámetros cargados:")
    print(f"  - Umbral temperatura: {params['thresholds']['temperature']}°C")
    print(f"  - Umbral CPU: {params['thresholds']['cpu_usage']}%")
    print(f"  - Umbral GPU: {params['thresholds']['gpu_usage']}%")
    print(f"  - Drop performance: {params['thresholds']['performance_drop']*100}%")
    
    # Cargar datos
    print("\nCargando datos...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"  - Train shape: {train_df.shape}")
    print(f"  - Test shape: {test_df.shape}")
    
    # Análisis estadístico
    print("\nEstadísticas del dataset de entrenamiento:")
    for col in ['temperature', 'cpu_usage', 'gpu_usage', 'performance']:
        print(f"  {col:15} - Mean: {train_df[col].mean():6.2f}, "
              f"Min: {train_df[col].min():6.2f}, Max: {train_df[col].max():6.2f}")
    
    # Crear variable target
    print("\nCreando variable target...")
    target, stats = create_target_variable(train_df, params)
    
    print(f"\nCondiciones críticas detectadas:")
    print(f"  - Temperatura crítica: {stats['temp_critica_pct']:.1f}%")
    print(f"  - CPU crítica: {stats['cpu_critica_pct']:.1f}%")
    print(f"  - GPU crítica: {stats['gpu_critica_pct']:.1f}%")
    print(f"  - Performance crítica: {stats['performance_critica_pct']:.1f}%")
    print(f"  - FALLOS TOTALES: {stats['fallos_pct']:.1f}%")
    
    # Guardar datos procesados
    print("\nGuardando datos procesados...")
    os.makedirs('data', exist_ok=True)
    
    train_processed = train_df.copy()
    train_processed['Fallos'] = target
    train_processed.to_csv('data/train_processed.csv', index=False)
    
    # Guardar estadísticas
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/preprocessing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("data/train_processed.csv")
    print("metrics/preprocessing_stats.json")
    
    # Crear visualizaciones
    print("\nGenerando visualizaciones...")
    create_visualizations(train_df, target)
    
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()