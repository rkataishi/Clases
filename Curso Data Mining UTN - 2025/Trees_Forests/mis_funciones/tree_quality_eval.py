# tree_quality_eval.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def tree_quality_eval(y_test, y_pred):
    print("\nIndicadores de calidad del modelo:")
    print("---------------------------------")

    # nRMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    std_y = np.std(y_test)
    nrmse = rmse / std_y
    print("nRMSE (RMSE / STD):", round(nrmse, 4))
    print("Interpretación: El error promedio representa un {:.1f}% de la variabilidad natural de las ventas".format(nrmse * 100))
    print("Un nRMSE < 1 indica que el modelo logra explicar parte de la variación usando las variables predictoras")

    # MASE
    mae_modelo = mean_absolute_error(y_test, y_pred)
    mae_naive = mean_absolute_error(y_test, np.full_like(y_test, np.mean(y_test)))
    mase = mae_modelo / mae_naive
    print("\nMASE (MAE modelo / MAE naive):", round(mase, 4))
    if mase < 1:
        print("Interpretación: El modelo mejora en un {:.1f}% respecto a predecir la media de y_test".format((1 - mase) * 100))
        print("O: El uso de mis features (variables X) mejora en un {:.1f}% la predicción respecto al error MAE de la media de y_test".format((1 - mase) * 100))
    else:
        print("Interpretación: El modelo es peor que simplemente predecir la media porque el error es mayor a 1")

    # SMAPE
    smape = np.mean(np.abs(y_test - y_pred) / ((np.abs(y_test) + np.abs(y_pred)) / 2)) * 100
    print("\nSMAPE (%):", round(smape, 2))
    print("Interpretación: En promedio, la diferencia entre predicción y valor real es del {:.1f}%".format(smape))
    print("O: Frente a una predicción perfecta (0% de error), el error del modelo es del {:.1f}%".format(smape))
    print("Nota: Si el error = 100%, el error promedio igualó al tamaño del fenómeno que se quería predecir. El modelo es inútil.")