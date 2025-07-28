
def sort_dataframe_columns(df):
    """
    Ordena las columnas de un DataFrame agrupando por prefijos similares.
    
    Args:
        df: DataFrame de pandas a ordenar
        
    Returns:
        DataFrame con columnas reordenadas
    """
    # Obtener nombres de columnas actuales
    columnas_actuales = df.columns.tolist()
    
    # Crear diccionario para agrupar columnas por prefijos
    prefijos_orden = {
        'RAZONSOCIAL': [],
        'CUIT': [],
        'RAMA': [],
        'PRECIOUNITARIO': [],
        'CANTIDAD': [],
        'FOBDOLAR': [],
        'CIFDOLAR': [],
        'FLETEDOLAR': [],
        'SEGURODOLAR': [],
        'KILOGRAMOS': [],
        'BASEIMPONIBLE': [],
        'CONDICION': [],
        'TIPODESPACHO': [],
        'ESTADO': [],
        'LOCALIDAD': [],
        'PUERTO': [],
        'PAIS': [],
        'FECHA': [],
        'MES': [],
        'ANIO': [],
        'TRIMESTRE': [],
        'NUMERO': [],
        'ITEM': [],
        'POSICION': [],
        'UNIDAD': [],
        'MEDIO': [],
        'des_': []
    }
    
    # Agrupar columnas por prefijos
    for col in columnas_actuales:
        asignada = False
        for prefijo in prefijos_orden.keys():
            if col.startswith(prefijo):
                prefijos_orden[prefijo].append(col)
                asignada = True
                break
        
        # Si no se asignó a ningún prefijo conocido, poner al final
        if not asignada:
            if 'OTROS' not in prefijos_orden:
                prefijos_orden['OTROS'] = []
            prefijos_orden['OTROS'].append(col)
    
    # Construir lista final de columnas ordenadas
    columnas_ordenadas = []
    for prefijo, columnas in prefijos_orden.items():
        if columnas:  # Solo agregar si hay columnas para este prefijo
            columnas_ordenadas.extend(columnas)
    
    # Reordenar el DataFrame
    df_ordenado = df[columnas_ordenadas]
    
    print(f"DataFrame reordenado con {len(df_ordenado.columns)} columnas")
    print("Primeras 10 columnas:", df_ordenado.columns[:10].tolist())
    print("Últimas 10 columnas:", df_ordenado.columns[-10:].tolist())
    
    return df_ordenado
