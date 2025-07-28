
import pandas as pd
from tabulate import tabulate
from IPython.display import display


def describe_vars_txt(df):
    # Check if DataFrame has columns
    if df.empty or len(df.columns) == 0:
        print("Error: El DataFrame está vacío o no tiene columnas")
        return None
        
    column_summary = pd.DataFrame({
        'Columna': df.columns,
        'Tipo': [df[col].dtype for col in df.columns],
        'Nulos': [df[col].isna().sum() for col in df.columns],
        'Únicos': [df[col].nunique() for col in df.columns],
    })

    def sample_values(col):
        if df[col].nunique() <= 10:
            return df[col].dropna().unique()[:5]
        return None

    column_summary['Valores_Muestra'] = [sample_values(col) for col in df.columns]

    # Store current display settings
    current_display = {
        'max_columns': pd.get_option('display.max_columns'),
        'max_rows': pd.get_option('display.max_rows'),
        'width': pd.get_option('display.width'),
        'max_colwidth': pd.get_option('display.max_colwidth')
    }

    # Set display options for wider output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None) 
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Mostrar con pandas (Jupyter-friendly)
    print(column_summary)
    print(df.describe().round(2).T)

    # Mostrar como tabla tabulada en texto plano, con ancho máximo
    print(tabulate(column_summary, headers='keys', tablefmt='github', showindex=False, maxcolwidths=None))

    # Restore original display settings
    for option, value in current_display.items():
        pd.set_option(f'display.{option}', value)

    return None