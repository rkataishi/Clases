
import pandas as pd
from tabulate import tabulate
from IPython.display import display

def describe_vars(df):
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

    # Mostrar con pandas (Jupyter-friendly)
    display(column_summary)
    display(df.describe().round(2).T)

    # Mostrar como tabla tabulada en texto plano
    print(tabulate(column_summary, headers='keys', tablefmt='github', showindex=False))

    return None