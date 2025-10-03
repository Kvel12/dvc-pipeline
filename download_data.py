#!/usr/bin/env python3
"""
Script para descargar datos de la competencia de Kaggle usando la API
"""
import os
import zipfile
import subprocess
import sys

def download_kaggle_competition(competition_name, output_dir="data"):
    """
    Descarga los datos de una competencia de Kaggle
    
    Args:
        competition_name (str): Nombre de la competencia
        output_dir (str): Directorio donde guardar los datos
    """
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Comando para descargar la competencia
        cmd = [
            "kaggle", "competitions", "download", 
            "-c", competition_name,
            "-p", output_dir
        ]
        
        print(f"Descargando competencia: {competition_name}")
        print(f"Comando: {' '.join(cmd)}")
        
        # Ejecutar el comando
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Descarga completada exitosamente")
            
            # Buscar archivos zip y descomprimirlos
            for filename in os.listdir(output_dir):
                if filename.endswith('.zip'):
                    zip_path = os.path.join(output_dir, filename)
                    print(f"Descomprimiendo: {filename}")
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    
                    # Eliminar el archivo zip después de descomprimir
                    os.remove(zip_path)
                    print(f"✅ {filename} descomprimido y eliminado")
            
            # Listar archivos descargados
            print("\nArchivos descargados:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
        else:
            print("Error en la descarga:")
            print(result.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    competition_name = "devices-error-detection-acmud-2025"
    download_kaggle_competition(competition_name)
    print("Proceso completado!")