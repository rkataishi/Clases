import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats

def linear_regression_detailed(target, predictors, X_scaled_df):
    """
    Realiza una regresión lineal detallada con estadísticos completos.
    
    Parameters:
    -----------
    target : str
        Nombre de la variable dependiente
    predictors : list
        Lista de nombres de variables predictoras
    X_scaled_df : pd.DataFrame
        DataFrame con las variables estandarizadas
    
    Returns:
    --------
    tuple : (model_stats, resultado_final)
        model_stats: DataFrame con estadísticos del modelo
        resultado_final: DataFrame con coeficientes ordenados por significancia
    """
    
    model = LinearRegression()
    model.fit(X_scaled_df[predictors], X_scaled_df[target])
    
    # Calcular predicciones
    y_pred = model.predict(X_scaled_df[predictors])
    y_true = X_scaled_df[target]
    
    # Calcular R²
    r2 = r2_score(y_true, y_pred)
    
    # Calcular p-values y estadísticos t
    n = len(y_true)
    p = len(predictors)
    residuals = y_true - y_pred
    mse = np.sum(residuals**2) / (n - p - 1)
    
    # Matriz X
    X_matrix = X_scaled_df[predictors].values
    X_with_intercept = np.column_stack([np.ones(n), X_matrix])
    
    # Calcular varianzas de los coeficientes
    var_beta = mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)).diagonal()
    se_beta = np.sqrt(var_beta[1:])  # Excluir intercepto
    
    # Calcular estadísticos t y p-values
    t_stats = model.coef_ / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-p-1))
    
    # Función para agregar estrellas según p-value
    def add_stars(p_val):
        if p_val < 0.001:
            return '***'
        elif p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        else:
            return ''
    
    # Crear DataFrame con todos los estadísticos
    coef_df = pd.DataFrame({
        'Variable': predictors,
        'Beta': model.coef_,
        'Std_Error': se_beta,
        't_stat': t_stats,
        'p_value': p_values,
        'Significance': [add_stars(p) for p in p_values]
    })
    
    # Crear DataFrame con estadísticos del modelo
    model_stats = pd.DataFrame({
        'Métrica': ['Variable Dependiente', 'R²', 'R² ajustado', 'N', 'Grados de libertad'],
        'Valor': [target, f"{r2:.4f}", f"{1 - (1-r2)*(n-1)/(n-p-1):.4f}", str(n), str(n-p-1)]
    })
    
    # Configurar pandas para mostrar solo 4 decimales
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Mostrar estadísticos del modelo
    display(model_stats)
    
    # Mostrar tabla ordenada por valor absoluto de beta
    resultado_final = coef_df.sort_values(by='Beta', key=abs, ascending=False)
    display(resultado_final)
    
    return model_stats, resultado_final

# Ejemplo de uso:
# target = 'RAMA_str_ELECTRONICA_ELECTRODOM'
# predictors = [v for v in X_scaled_df.columns if v != target]
# model_stats, resultado_final = linear_regression_detailed(target, predictors, X_scaled_df)