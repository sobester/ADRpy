import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from .html_utils import convertir_a_html
from .imputation_utils import imprimir_detalles_imputacion





def imputacion_similitud_con_rango(df_filtrado, df_procesado, rango_min=0.85, rango_max=1.15, nivel_confianza_min_similitud=0.7):
    
    """
    Ajusta el rango de similitud e imputa valores faltantes en los parámetros de df_filtrado.
    Genera un reporte final en HTML con un resumen agregado, filtrando por nivel de confianza.
    :param df_filtrado: DataFrame con los parámetros a imputar.
    :param df_procesado: DataFrame procesado con todos los datos.
    :return: Nuevo DataFrame con los valores imputados.
    """
    print(f"Rango mínimo: {rango_min}, Rango máximo: {rango_max}, Confianza mínima: {nivel_confianza_min_similitud}")

    # Crear una copia para mantener intacto el DataFrame original
    df_resultado_por_similitud = df_filtrado.copy()          

    # Función interna para imputar por similitud
    def imputar_por_similitud(datos, parametro, aeronave, rango_min, rango_max, numero_valor_imputado):

        """
        Imputa un valor faltante basado en la similitud dentro del rango de MTOW.
        Detalla el proceso con mensajes informativos y utiliza la mediana para calcular el valor imputado.
        :param datos: DataFrame con los datos originales.
        :param parametro: Parámetro a imputar.
        :param aeronave: Aeronave con valor faltante.
        :param rango_min: Rango mínimo de similitud.
        :param rango_max: Rango máximo de similitud.
        :return: Tuple (valor imputado, nivel de confianza) o (None, None) si no es posible imputar.
        """

        if df_filtrado.isna().all().all():
            print("Todos los valores en 'df_filtrado' están vacíos. No se puede proceder con la imputación.")
            return df_filtrado, []
        
        if df_procesado.isna().all().all():
            print("Todos los valores en 'df_procesado' están vacíos. No se puede proceder con la imputación.")
            return df_filtrado, []

        try:
            # Verificar si MTOW está presente
            if "Peso máximo al despegue (MTOW)" not in datos.index:
                print(f"Advertencia: 'Peso máximo al despegue (MTOW)' no está en los datos para la aeronave '{aeronave}'.")
                return None, None
    
            # Obtener el valor actual de MTOW
            mtow_actual = datos.loc["Peso máximo al despegue (MTOW)", aeronave]
    
            # Validar si MTOW es válido
            if pd.isna(mtow_actual):
                print(f"MTOW faltante para la aeronave '{aeronave}'. Imputación no es posible.")
                return None, None
    
            # Filtrar candidatas dentro del rango ajustado
            candidatas_mtow = datos.loc[
                :, (datos.loc["Peso máximo al despegue (MTOW)"] >= rango_min * mtow_actual) &
                   (datos.loc["Peso máximo al despegue (MTOW)"] <= rango_max * mtow_actual)
            ]
    
            if candidatas_mtow.empty:
                print(f"No hay candidatos dentro del rango de {rango_min*100:.0f}% - {rango_max*100:.0f}%.")
                return None, None
    
            # Excluir la misma aeronave de las candidatas
            candidatas_mtow = candidatas_mtow.loc[:, candidatas_mtow.columns != aeronave]
    
            # Verificar que las candidatas tengan valores válidos en el parámetro a imputar
            candidatas_validas = candidatas_mtow.loc[parametro].dropna()
    
            if candidatas_validas.empty:
                print(f"Razón: Ninguna aeronave se encuentra dentro del rango MTOW de '{aeronave}'para el parametro '{parametro}'.")
                return None, None
    

            # Calcular los valores ajustados individualmente en función del MTOW de cada candidata
            valores_ajustados = []
            mtow_candidatos = []  # Nueva lista para almacenar los valores de MTOW de los candidatos
            detalles_ajustes = []
            
            for candidata in candidatas_validas.index:
                mtow_candidata = datos.loc["Peso máximo al despegue (MTOW)", candidata]
                relacion_mtow_individual = mtow_candidata / mtow_actual
                ajuste_individual = (relacion_mtow_individual - 1) / 4
            
                valor_candidata = candidatas_validas[candidata]
                valor_ajustado = valor_candidata * (1 + ajuste_individual)
            
                # Guardar en las listas correspondientes
                valores_ajustados.append(valor_ajustado)
                mtow_candidatos.append(mtow_candidata)

                
                # Registrar detalle del ajuste
                detalles_ajustes.append({
                    "Aeronave": candidata,
                    "MTOW Candidata": mtow_candidata,
                    "Relación MTOW": relacion_mtow_individual,
                    "Ajuste Individual": ajuste_individual,
                    "Valor Original": valor_candidata,
                    "Valor Ajustado": valor_ajustado
                })
                  

            # Usar la mediana de los valores ajustados como valor final imputado
            valor_imputado = np.median(valores_ajustados)
            
            # Calcular la confianza directamente
            cantidad_minima = 3
            penalizacion_candidatos = 1 - (cantidad_minima - len(valores_ajustados)) / (3 * cantidad_minima)
            penalizacion_candidatos = max(0, penalizacion_candidatos)  # Evitar valores negativos
            
            # Cantidad ponderada basada en calidad
            pesos_mtow = np.exp(-np.abs(np.array(mtow_candidatos) - mtow_actual) / mtow_actual)
            cantidad_ponderada = np.sum(pesos_mtow) / len(mtow_candidatos)
            
            # Evaluación de R^2 con normalización
            if len(candidatas_validas) > 1:
                try:
                    r2 = r2_score(candidatas_validas.tolist(), valores_ajustados)
                    ponderacion_modelo = r2
                except ValueError:
                    print("No se puede calcular R^2: insuficientes datos válidos.")
                    r2 = None
            else:
                r2 = None
            
            # Usar dispersión como respaldo cuando R^2 no se puede calcular
            if r2 is None:
                dispersion = np.std(valores_ajustados) if len(valores_ajustados) > 1 else 1.0
                ponderacion_modelo = 1 / (1 + dispersion)
            
            # Validar ponderación del modelo
            if ponderacion_modelo is not None and (ponderacion_modelo < 0 or ponderacion_modelo > 1):
                print(f"Advertencia: Ponderación del modelo ({ponderacion_modelo:.3f}) fuera de rango. Revisar lógica previa.")
                ponderacion_modelo = max(0, min(ponderacion_modelo, 1))  # Forzar a rango válido

            
            # Confianza final combinada
            peso_candidatos = 0.6
            peso_modelo = 0.4
            confianza_base = (
                peso_candidatos * cantidad_ponderada +
                peso_modelo * ponderacion_modelo
            )
            
            # Aplicar penalización por pocos candidatos
            confianza = confianza_base * penalizacion_candidatos
            
            # Asegurar valores entre 0 y 1
            confianza = min(1.0, max(0.0, confianza))

            # Detalles del cálculo de confianza
            calculos_confianza = {
                "Penalización por pocos candidatos": penalizacion_candidatos,
                "Cantidad Ponderada (basada en MTOW)": cantidad_ponderada,
                "Ponderación del modelo (R² o dispersión)": ponderacion_modelo,
                "Confianza Base": confianza_base,
                "Confianza Final (tras penalización)": confianza
            }

            imprimir_detalles_imputacion(
                numero_valor_imputado=numero_valor_imputado,
                parametro=parametro,
                aeronave=aeronave,
                mtow_actual=mtow_actual,
                rango_min=rango_min,
                rango_max=rango_max,
                candidatas_validas=candidatas_validas,
                detalles_ajustes=detalles_ajustes,
                valores_ajustados=valores_ajustados,
                valor_imputado=valor_imputado,
                confianza=confianza,
                calculos_confianza=calculos_confianza
            )


            return valor_imputado, confianza
                
            
        except Exception as e:
            print(f"Error durante la imputación: {e}")
            return None, None




    try:
        # Mostrar df_resultado_por_similitud en HTML para verificar valores iniciales
        print("\n=== Verificación de 'df_resultado_por_similitud' ===")
        convertir_a_html(df_resultado_por_similitud, titulo="Datos Filtrados por aeronaves seleccionadas antes de imputar(df_resultado_por_similitud)", mostrar=True)

        # Reporte de imputaciones
        reporte_imputaciones = []
        numero_valor_imputado = 0

        for parametro in df_resultado_por_similitud.index:
            for aeronave in df_resultado_por_similitud.columns:
                if pd.isna(df_resultado_por_similitud.loc[parametro, aeronave]):
                    # Realizar imputación
                    valor_imputado, confianza = imputar_por_similitud(
                        df_procesado, parametro, aeronave, rango_min, rango_max, numero_valor_imputado
                    )

                    # Verificar si se realizó una imputación válida
                    if valor_imputado is not None and confianza is not None and confianza >= nivel_confianza_min_similitud:
                        # Incrementar el contador SOLO aquí
                        numero_valor_imputado += 1
                        # Asignar el valor imputado al DataFrame
                        df_resultado_por_similitud.loc[parametro, aeronave] = valor_imputado
                        # Registrar la imputación en el resumen
                        reporte_imputaciones.append({
                            "Aeronave": aeronave,
                            "Parámetro": parametro,
                            "Valor Imputado": valor_imputado,
                            "Nivel de Confianza": confianza
                        })

                    elif confianza is not None and confianza < nivel_confianza_min_similitud:
                        print(f"Imputación descartada por baja confianza: {confianza:.3f} < {nivel_confianza_min_similitud}.")
                    else:
                        print(f"No se pudo imputar: {parametro} para {aeronave}.")



        # Generar reporte final en HTML con filtro de confianza
        print("\n=== Generando reporte final ===")
        if reporte_imputaciones:
            df_reporte = pd.DataFrame(reporte_imputaciones)
            #print("Contenido de reporte_imputaciones:", reporte_imputaciones)
            df_reporte = df_reporte[df_reporte["Nivel de Confianza"] >= nivel_confianza_min_similitud]
            convertir_a_html(df_reporte, titulo="Reporte Final de Imputaciones", mostrar=True)
        else:
            print("No se realizaron imputaciones con el nivel de confianza aceptable.")


    except Exception as e:
        print(f"Error durante la imputación: {e}")

    # Generar DataFrame con valores imputados
    if not reporte_imputaciones:
        print("No se realizaron imputaciones con éxito.")
        return df_filtrado, []
        

    # Convertir lista de diccionarios a DataFrame para exportación final
    df_reporte_final = pd.DataFrame(reporte_imputaciones)
    
    # Asegurar estructura del DataFrame imputado
    df_resultado_final = df_resultado_por_similitud.copy()
    
    # Validar resultados
    if df_resultado_final.empty:
        print("El DataFrame de resultados está vacío.")
        return df_filtrado, []
    
    # Retornar DataFrame imputado y lista de diccionarios
    return df_resultado_final, reporte_imputaciones

