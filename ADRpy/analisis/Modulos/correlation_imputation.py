import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .html_utils import convertir_a_html
from .correlation_analysis import calcular_correlaciones_y_generar_heatmap_con_resumen




def Imputacion_por_correlacion(
    df_correlacion,
    parametros_preseleccionados,
    tabla_completa,
    parametros_seleccionados,
    min_datos_validos=5,
    max_lineas_consola=250,
    umbral_correlacion=0.7,
    nivel_confianza_min_correlacion=0.5,
):
    # Lógica de la función
        """
        Imputa valores faltantes en un dataframe basado en correlaciones significativas entre parámetros.
    
        Parámetros:
            df_procesado (pd.DataFrame): DataFrame con los datos a procesar.
            parametros_preseleccionados (list): Lista de parámetros a imputar.
            umbral_correlacion (float): Valor mínimo absoluto de correlación para considerar como significativa.
            min_datos_validos (int): Mínimo número de datos válidos requeridos por parámetro para ser incluido.
            max_lineas_consola (int): Máximo número de líneas a imprimir en la consola.
    
        Retorna:
            pd.DataFrame: DataFrame con valores imputados.
        """
        # Cargar datos simulados
        df = df_correlacion.copy()

        if tabla_completa is None or tabla_completa.empty:
            raise ValueError("La tabla de correlaciones completa no fue proporcionada o está vacía.")

       # Mostrar df en formato HTML
        print("\n=== DataFrame inicial ===")
        convertir_a_html(df, titulo="DataFrame antes de realizar imputacion por correlacion (df_procesado.copy())", mostrar=True)
        #print("Parámetros disponibles en el índice del DataFrame:")
        #print(df.index.tolist())
    
        # Convertir todo a numérico
        print("\n=== Convertir todo a numérico ===")
        df = df.apply(pd.to_numeric, errors='coerce')  # Forzar datos no numéricos a NaN
        
        #afasfasfasfasf no se que hace
        parametros_validos = df.index[df.notna().sum(axis=1) >= 5].tolist()
        df = df.loc[parametros_validos]
        print(type(parametros_validos))
    

        # === PASO 1: CÁLCULO DE CORRELACIONES ===
        print("\n=== PASO 1: CÁLCULO DE CORRELACIONES ENTRE PARÁMETROS ===")
        tabla_completa = tabla_completa = calcular_correlaciones_y_generar_heatmap_con_resumen(
    df,
    parametros_seleccionados,
    umbral_heat_map=umbral_correlacion,
    devolver_tabla=True
)

        correlaciones = tabla_completa.copy()
        indices_validos = df.index
    
        # Filtrar correlaciones para que coincidan los índices y las columnas con parámetros válidos
        correlaciones_filtradas = correlaciones.loc[
            indices_validos.intersection(correlaciones.index),  # Filtra filas válidas
            indices_validos.intersection(correlaciones.columns)  # Filtra columnas válidas
        ]

        correlaciones_aceptables = correlaciones_filtradas[(correlaciones_filtradas.abs() >= 0.7) & (correlaciones_filtradas.abs() < 1.0)]
        
        # Mostrar tabla de correlaciones
        #convertir_a_html(correlaciones, titulo="Tabla de Correlaciones", mostrar=True)
        #print("Parámetros disponibles en el índice del DataFrame:")
        #print(correlaciones.index.tolist())
    
        # Mostrar correlaciones aceptables en HTML
        convertir_a_html(
            datos_procesados=correlaciones_aceptables,
            titulo="Tabla de correlaciones con filtro de umbral de correlación",
            mostrar=True
        )
        #print("Parámetros disponibles en el índice del DataFrame:")
        #print(correlaciones_aceptables.index.tolist())  
        
        # === PASO 2: IMPUTACIÓN ===
        print("\n=== PASO 2: IMPUTACIÓN DE VALORES ===")
        valores_imputados = 0
        lineas_impresas = 0
        MAX_LINEAS_CONSOLA = 40000000
        
        def evaluar_confianza(puntaje):
            """Evalúa el nivel de confianza basado en el puntaje."""
            if puntaje >= 0.9:
                return "Confianza Muy Alta"
            elif puntaje >= 0.75:
                return "Confianza Alta"
            elif puntaje >= 0.6:
                return "Confianza Media"
            elif puntaje >= 0.4:
                return "Confianza Baja"
            else:
                return "Confianza Muy Baja"


        #Declara una variable para crear una lista para registrar las imputaciones
        reporte_imputaciones = []


    
        for parametro in parametros_preseleccionados:
            if parametro not in correlaciones_aceptables.columns:  # Cambiar lógica para trabajar con columnas
                print(f"\n=== {parametro}: Sin correlaciones significativas (|r| < 0.7) ===")
                continue

            valores_faltantes = df[parametro][df[parametro].isna()].index.tolist()  # Ajustar para trabajar con columnas
            if not valores_faltantes:
                print(f"\n=== {parametro}: No hay valores faltantes para imputar. ===")
                continue

            print(f"\n=== Imputación para el parámetro: **{parametro}** ===")
            for aeronave in valores_faltantes:
                if lineas_impresas >= MAX_LINEAS_CONSOLA:
                    print("\n--- Límite de impresión alcanzado. ---")
                    break

                print(f"\n--- Imputación para aeronave: **{aeronave}** ---")
                valores_predichos = []

                correlaciones_parametro = correlaciones_aceptables[parametro].dropna()  # Ajustar para trabajar con columnas

                for parametro_correlacionado, correlacion in correlaciones_parametro.items():
                    datos_validos = df[[parametro, parametro_correlacionado]].dropna(axis=0)  # Ajustar para trabajar con filas

                    if datos_validos.shape[0] < 5:  # Cambiar a filas
                        continue

                    # Evitar duplicados
                    datos_validos = datos_validos.drop_duplicates()

                    X = datos_validos[parametro_correlacionado].values.reshape(-1, 1)
                    y = datos_validos[parametro].values

                    # Entrenar modelo de regresión
                    modelo = LinearRegression().fit(X, y)
                    r2 = modelo.score(X, y)
                    desviacion_std = np.std(y - modelo.predict(X))
                    varianza = np.var(y - modelo.predict(X))
                    incertidumbre = desviacion_std / np.sqrt(len(y))
                    puntaje_confianza = 0.4 * r2 + 0.3 * (1 - incertidumbre) + 0.2 * (1 - desviacion_std) + 0.1 * (1 - varianza)
                    nivel_confianza = evaluar_confianza(puntaje_confianza)

                    if pd.notna(df.at[aeronave, parametro_correlacionado]):  # Ajustar para trabajar con filas
                        valor_imputado = modelo.predict(np.array([[df.at[aeronave, parametro_correlacionado]]]))[0]  # Convertir a numpy.ndarray
                        valores_predichos.append(
                            (parametro_correlacionado, round(valor_imputado, 3), round(r2, 3), round(desviacion_std, 3))
                        )

                        # Detalle de datos usados
                        print(f"\n--- Correlación: {parametro_correlacionado} (r = {round(correlacion, 3)}) ---")
                        print(f"Aeronaves utilizadas: {datos_validos.index.tolist()}")
                        print(f"Valores para {parametro_correlacionado}: {X.flatten().round(3).tolist()}")
                        print(f"Valores para {parametro}: {y.round(3).tolist()}")
                        print(f"Ecuación de regresión: y = {round(modelo.coef_[0], 3)}x + {round(modelo.intercept_, 3)}")
                        print(f"Valor del parámetro correlacionado para la aeronave: {round(df.at[aeronave, parametro_correlacionado], 3)}")
                        print(f"Predicción obtenida: {round(valor_imputado, 3)}")
                        print(f"\tR²: {r2}, Desviación Estándar: {desviacion_std}, Varianza: {varianza}, Incertidumbre: {incertidumbre}")
                        print(f"\tNivel de confianza: {nivel_confianza}")
                        lineas_impresas += 1

                if valores_predichos:
                    valor_final = np.median([pred[1] for pred in valores_predichos])
                    df.at[aeronave, parametro] = round(valor_final, 3)  # Ajustar para trabajar con filas
                    valores_imputados += 1
                    print(f"Valores imputados: {[f'{pred[0]}: {pred[1]}' for pred in valores_predichos]}")
                    print(f"**Mediana calculada:** {round(valor_final, 3)}")

                    # Registro correcto
                    reporte_imputaciones.append({
                        "Aeronave": aeronave,
                        "Parámetro": parametro,
                        "Valor Imputado": valor_final,
                        "Nivel de Confianza": puntaje_confianza
                    })
                else:
                    info_imposible = pd.DataFrame([{
                        "Mensaje": f"No se pudo imputar el parámetro '{parametro}' para la aeronave '{aeronave}'."
                    }])
                    convertir_a_html(info_imposible, titulo="Imputación no Exitosa", mostrar=True)

                    lineas_impresas += 1


        # Filtro y generación del reporte final
        df_reporte = pd.DataFrame(reporte_imputaciones)
        #print("Contenido de reporte_imputaciones:", reporte_imputaciones)
        #print("Columnas de df_reporte:", df_reporte.columns)
        #print("Contenido inicial de df_reporte:\n", df_reporte.head())
        if "Nivel de Confianza" in df_reporte.columns:
            df_reporte = df_reporte[df_reporte["Nivel de Confianza"] >= nivel_confianza_min_correlacion]
        else:
            print("La columna 'Nivel de Confianza' no está presente en df_reporte.")
            # Maneja el caso sin filtro, por ejemplo:
            return df, []

    
        # Resumen de imputaciones
        resumen_imputaciones = df_reporte.groupby("Aeronave").size().reset_index(name="Cantidad de Valores Imputados")
        total_imputaciones = resumen_imputaciones["Cantidad de Valores Imputados"].sum()
        resumen_imputaciones.loc["Total"] = ["Total", total_imputaciones]
    
        # Mostrar reportes (HTML opcional)
        convertir_a_html(df_reporte, titulo="Reporte Final de Imputaciones", mostrar=True)
        convertir_a_html(resumen_imputaciones, titulo="Resumen de Imputaciones", mostrar=True)

        # Validar si se realizaron imputaciones
        if not reporte_imputaciones:
            print("No se realizaron imputaciones con éxito.")
            return df, []
        
        # Validar si el DataFrame está vacío (seguridad adicional)
        if df.empty:
            print("El DataFrame de resultados está vacío.")
            return df, []
        
        # Opcional: convertir reporte_imputaciones a DataFrame si necesario
        #df_reporte_final = pd.DataFrame(reporte_imputaciones)
        
        # Retornar el DataFrame procesado y la lista de imputaciones
        return df, reporte_imputaciones