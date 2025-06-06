"""
imputacion_similitud_flexible.py
--------------------------------
Implementa la lógica de K‑NN con 3 ejes obligatorios (físico, geométrico, prestacional)
y filtrado progresivo de familia (F0‑F3).  Diseñado para integrarse sin romper
los nombres ni los flujos que ya existen en tu proyecto ADRpy.

Uso rápido:
    python imputacion_similitud_flexible.py --ruta_excel Datos_aeronaves.xlsx \
        --aeronave "Stalker XE" \
        --parametro "Velocidad a la que se realiza el crucero (KTAS)"
"""

import argparse
import pandas as pd
import numpy as np
# ------------------------ HELPERS ------------------------

def imprimir(msg, bold=False):
    prefix = "\033[1m" if bold else ""
    suffix = "\033[0m" if bold else ""
    print(f"{prefix}{msg}{suffix}")
# ------------------------------------------------------------------ #
#  CONFIGURACIÓN DE BLOQUES Y CAPAS DE FAMILIA
# ------------------------------------------------------------------ #

def configurar_similitud():
    """
    Devuelve:
    - bloques_rasgos: diccionario de ejes y parámetros
    - filas_familia: atributos para clasificación familiar
    - capas_familia: filtros progresivos de familia F0, F1, F2
    """
    bloques_rasgos = {
        "fisico": ["Peso máximo al despegue (MTOW)", "Peso vacío"],
        "geom": ["Área del ala", "envergadura", "Longitud del fuselaje", "Relación de aspecto del ala"],
        "prest": [
            "Potencia específica (P/W)", "Autonomía de la aeronave", 
            "Alcance de la aeronave", "Velocidad a la que se realiza el crucero (KTAS)",
            "Velocidad máxima (KIAS)",
        ],
    }
    filas_familia = [
        "Misión", "Despegue", "Propulsión vertical", "Propulsión horizontal",
        "Cantidad de motores propulsión vertical", "Cantidad de motores propulsión horizontal",
    ]
    capas_familia = [
        {attr: "equals" for attr in filas_familia},
        {attr: "equals" for attr in filas_familia[:4]},
        {attr: "equals" for attr in filas_familia[:2]},
    ]
    return bloques_rasgos, filas_familia, capas_familia

# ------------------------ FUNCIÓN PRINCIPAL ------------------------

def penalizacion_por_k(k):
    """
    Calcula la penalización por cantidad de vecinos (k) usando una ecuación polinomial.
    Para k > 10, la penalización se fija en 1.0 (confianza máxima).
    """
    if k > 10:
        return 1.0
    return max(0, min(1, 0.00002281 * k**5 - 0.00024 * k**4 - 0.0036 * k**3 + 0.046 * k**2 + 0.0095 * k + 0.024))

def imputar_por_similitud(
    df_parametros: pd.DataFrame,
    df_atributos: pd.DataFrame,
    aeronave_obj: str,
    parametro_objetivo: str,
    bloques_rasgos: dict,
    capas_familia: list
):
    imprimir(f"\n=== Imputación por similitud: {aeronave_obj} - {parametro_objetivo} ===", True)

    # Validaciones
    if parametro_objetivo not in df_parametros.columns:
        imprimir(f"⚠️ Parámetro '{parametro_objetivo}' no encontrado.", True)
        return None
    if aeronave_obj not in df_parametros.index:
        imprimir(f"⚠️ Aeronave '{aeronave_obj}' no encontrada.", True)
        return None

    # Iteración por capas de familia
    for capa_idx, criterios in enumerate(capas_familia):
        familia = f"F{capa_idx}"
        imprimir(f"\n--- Capa {familia}: criterios {list(criterios.keys())} ---", True)
        # Filtrar familia
        mask = np.ones(df_parametros.shape[0], dtype=bool)
        for fila, modo in criterios.items():
            val = df_atributos.loc[aeronave_obj, fila]
            mask &= (df_atributos[fila] == val).values
        df_familia = df_parametros.loc[mask, :]
        if df_familia.shape[0] == 0:
            imprimir(f"❌ Sin drones en {familia}. Relajando...", True)
            continue
        #  —> Validar que haya vecinos con el parámetro objetivo
        rows_validas = df_familia.index[df_familia[parametro_objetivo].notna()]
        if len(rows_validas) == 0:
            imprimir(f"❌ Ningún dron en {familia} tiene '{parametro_objetivo}'.", True)
            continue

        # Parámetros MTOW y filtro ±20%
        mtow_obj = df_familia.loc[aeronave_obj, "Peso máximo al despegue (MTOW)"]
        mtow_vec = df_familia.loc[rows_validas, "Peso máximo al despegue (MTOW)"].values
        delta_mtow = np.abs(mtow_vec - mtow_obj) / mtow_obj * 100
        mask_mtow = delta_mtow <= 20
        rows_filtrados = rows_validas[mask_mtow]
        if len(rows_filtrados) == 0:
            imprimir(f"❌ Sin vecinos ±20% MTOW en {familia}.", True)
            continue

        # Cálculo de MTOW_score
        d = delta_mtow[mask_mtow]
        g = -0.002*d**4 + 0.041*d**3 - 0.28135*d**2 + 0.23*d + 99.94
        mtow_scores = g / 100.0

        # Nueva lógica para calcular los bonos geométricos y prestacionales

        def calcular_bono(tipo):
            parametros = bloques_rasgos[tipo]  # Obtener los parámetros geométricos o prestacionales
            bono_total = 0  # Inicializar el bono total

            for parametro in parametros:
                try:
                    # Valores de la aeronave objetivo y los vecinos
                    valor_objetivo = df_parametros.loc[aeronave_obj, parametro]
                    valores_vecinos = df_familia.loc[rows_filtrados, parametro].values

                    # Si el valor de la aeronave objetivo es NaN, el bono es 0
                    if pd.isna(valor_objetivo):
                        imprimir(f"⚠️ Parámetro '{parametro}' no tiene valor en la aeronave objetivo. Bono = 0.")
                        continue

                    # Calcular las diferencias relativas para los vecinos válidos
                    diferencias = np.abs(valores_vecinos - valor_objetivo) / valor_objetivo * 100

                    for d, vecino in zip(diferencias, valores_vecinos):
                        if pd.isna(d):
                            imprimir(f"⚠️ Diferencia NaN para el parámetro '{parametro}'. Vecino: {vecino}. Bono = 0.")
                            continue

                        # Ajustar la diferencia relativa según el rango
                        if d > 40:
                            g = -100  # Máximo bono negativo
                        elif d > 20:
                            # Recalcular la diferencia relativa en el rango 20% a 40%
                            d_ajustada = d - 20
                            g = -0.002 * d_ajustada**4 + 0.041 * d_ajustada**3 - 0.28135 * d_ajustada**2 + 0.23 * d_ajustada + 99.94
                            g = -g  # Cambiar el signo a negativo
                        else:
                            # Rango de 1% a 20%
                            g = -0.002 * d**4 + 0.041 * d**3 - 0.28135 * d**2 + 0.23 * d + 99.94

                        # Calcular el bono para este parámetro y vecino
                        bono_parametro = (g / 100) * 0.05
                        bono_total += bono_parametro

                        # Imprimir detalles para depuración
                        imprimir(f"  Parámetro: {parametro}, Vecino: {vecino}, Objetivo: {valor_objetivo}, d: {d:.2f}%, g: {g:.2f}, Bono: {bono_parametro:.5f}")
                        
                except KeyError:
                    imprimir(f"⚠️ Parámetro '{parametro}' no encontrado en los datos. Ignorando.")
                    continue

            imprimir(f"  Bono total para '{tipo}': {bono_total:.5f}")
            return bono_total

        # Calcular los bonos geométricos y prestacionales
        bonus_geom = calcular_bono("geom")
        bonus_prest = calcular_bono("prest")
        imprimir(f"  Bono geométrico: {bonus_geom:.3f}")
        imprimir(f"  Bono prestacional: {bonus_prest:.3f}")

        # Score de familia
        family_scores = {0: 0.95, 1: 0.825, 2: 0.70}
        fam_score = family_scores[capa_idx]
        sim_i = fam_score * mtow_scores + bonus_geom + bonus_prest

        # Mostrar similitudes
        for nbr, s in zip(rows_filtrados, sim_i):
            imprimir(f" vecino '{nbr}' → sim_i: {s:.3f}")

        # Filtrar por umbral
        umbral = 0.0
        mask_sim = sim_i >= umbral
        vecinos_val = rows_filtrados[mask_sim]
        sim_vals = sim_i[mask_sim]
        if len(vecinos_val) == 0 or sim_vals.sum() < 1e-6:
            imprimir(f"❌ Sin vecinos ≥{umbral} en {familia}.", True)
            continue

        # Imputación y confianza
        y = df_familia.loc[vecinos_val, parametro_objetivo].values
        valor_imp = np.dot(sim_vals, y) / sim_vals.sum()

        # Cálculo de métricas estadísticas
        if len(y) > 1:
            media_y = np.mean(y)
            dispersion = np.std(y, ddof=0)
            cv = dispersion / media_y if media_y != 0 else 0  # Coeficiente de variación
        else:
            media_y = y[0] if len(y) == 1 else 0  # Si hay un solo valor, usarlo; si no hay valores, asignar 0
            cv = 1  # Penalización máxima para k=1
            dispersion = 0

        # Penalización por cantidad de datos usados
        penalizacion_k = penalizacion_por_k(len(vecinos_val))
        # Penalización por la calidad de los datos
        confianza_cv = max(0,1-(cv/0.5)) #cuando la desviacion estandar es igual al 50% de la media entonces confianza 0  # Dispersión de los valores

        # Confianza final combinada
        beta = 0.7  # Peso para cantidad de datos usados
        pesos = sim_vals / sim_vals.sum()  # Normalizar los pesos
        promedio_sim_i = np.dot(pesos, sim_vals)
        confianza_datos = beta * penalizacion_k + (1 - beta) * (confianza_cv) # Confianza basada en K y CV
        confianza_final = promedio_sim_i*confianza_datos  # Confianza final = promedio de confianza de vecinos * confianza de datos

        # Mostrar detalles del cálculo de confianza final
        imprimir("\nDetalles del cálculo de confianza:")

        imprimir(f"  Confianza que tan similares son los vecinos (familia x Mtow + Bonos): {[f'{s:.3f}' for s in sim_i]}")
        imprimir(f"  Promedio ponderado de confianza de similitud de aeronaves: {promedio_sim_i:.3f}")
        imprimir(f"  Media de valores (y): {media_y:.3f}")
        imprimir(f"  Coeficiente de variación (CV): {confianza_cv:.3f}")
        imprimir(f"  Dispersión: {dispersion:.3f}")
        imprimir(f"  Penalización por cantidad de vecinos (k): {penalizacion_k:.3f}")        
        imprimir(f"  Confianza en base a la calidad y cantidad de datos: {confianza_datos:.3f}")
        imprimir(f"  Confianza final: {confianza_final*100:.3f}%")


        # Retornar resultados
        imprimir(f"✅ Valor imputado: {valor_imp:.3f} (conf {confianza_final:.3f}, datos {len(vecinos_val)}, familia {familia})")

        return {
            "valor": valor_imp,
            "confianza": confianza_final,
            "num_vecinos": len(vecinos_val),
            "familia": familia
        }

    imprimir("⚠️ No se pudo imputar en ninguna capa. Delegar a correlación...", True)
    return None
