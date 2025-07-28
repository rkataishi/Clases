import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def analyze_variable(df, var_name):
    """
    Analiza una variable del DataFrame mostrando estadísticas descriptivas, 
    frecuencia y visualización.
    
    Parameters:
    df: DataFrame
    var_name: str, nombre de la variable a analizar
    """
    # Estadísticas descriptivas
    print(f"\nEstadísticas de {var_name}:")
    print(f"Total de observaciones: {len(df)}")
    print(f"Valores únicos: {df[var_name].nunique()}")
    print(f"Rango: {df[var_name].min()} - {df[var_name].max()}")
    print("\n")
    
    # Frecuencia y distribución porcentual
    var_freq = df[var_name].value_counts().sort_index()
    var_pct = (df[var_name].value_counts(normalize=True).sort_index() * 100).round(2)
    
    # Crear DataFrame para mostrar la información
    summary_df = pd.DataFrame({
        var_name: var_freq.index,
        'Frecuencia': var_freq.values,
        'Porcentaje': var_pct.values
    }).sort_values('Frecuencia', ascending=False)
    display(summary_df)
    # Eliminar el DataFrame de resumen después de mostrarlo
    del summary_df
    
    
    # Visualización de la distribución
    plt.figure(figsize=(12, 6))
    df[var_name].value_counts().sort_index().plot(kind='barh')
    plt.title(f'Distribución de {var_name}')
    plt.ylabel(var_name)
    plt.xlabel('Frecuencia')
    plt.tight_layout()
    plt.show()