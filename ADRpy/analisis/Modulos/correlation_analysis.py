import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .html_utils import convertir_a_html
from .user_interaction import solicitar_umbral


def calcular_correlaciones_y_generar_heatmap_con_resumen(df_procesado, parametros_seleccionados, umbral_heat_map=None, devolver_tabla=False):
    valor_por_defecto = 0.7
    """
    Calcula las correlaciones completas y filtradas entre variables seleccionadas,
    genera tablas en HTML con un resumen agregado, y crea un heatmap.
    :param df_procesado: DataFrame procesado con los datos completos.
    :param parametros_seleccionados: Lista de variables a incluir en los cálculos y visualización.
    :param valor_por_defecto: Umbral predeterminado para correlaciones significativas.
    :param devolver_tabla: Si True, retorna la tabla completa de correlaciones.
    :return: Tabla completa de correlaciones (opcional).
    """

    def agregar_resumen_a_tabla(tabla, titulo):
        """
        Agrega un resumen al final de una tabla HTML indicando:
        - Cantidad total de valores.
        - Cantidad de valores numéricos.
        - Cantidad de valores NaN.
        """
        total_valores = tabla.size
        valores_numericos = tabla.count().sum()
        valores_nan = total_valores - valores_numericos

        resumen = pd.DataFrame(
            {
                "Resumen": ["Total de valores", "Valores numéricos", "Valores NaN"],
                "Cantidad": [total_valores, valores_numericos, valores_nan],
            }
        )

        convertir_a_html(tabla, titulo=titulo, mostrar=True)
        convertir_a_html(resumen, titulo="Resumen de la Tabla", mostrar=True)

    try:
        # === Paso 1: Obtener umbral_heat_map desde el argumento o pedirlo al usuario ===
        if umbral_heat_map is None:
            umbral_heat_map = solicitar_umbral(valor_por_defecto)

        print(f"\nUmbral seleccionado para correlaciones significativas: {umbral_heat_map}")

        # === Validación de parámetros seleccionados ===
        parametros_no_encontrados = [
            v for v in parametros_seleccionados if v not in df_procesado.columns
        ]
        if parametros_no_encontrados:
            raise ValueError(
                f"Los siguientes parámetros no se encontraron en los datos procesados: {', '.join(parametros_no_encontrados)}"
            )

        # === Tabla completa (sin filtrar) ===
        print("\n=== Cálculo de tabla completa ===")
        tabla_completa = df_procesado.corr()
        agregar_resumen_a_tabla(
            tabla_completa.round(3),
            "Tabla de Correlaciones con todos los parametros(tabla_completa)",
        )

        # Filtrar correlaciones por el umbral_heat_map
        tabla_completa_significativa = tabla_completa[
            (tabla_completa.abs() >= umbral_heat_map) & (tabla_completa != 1)
        ]
        # agregar_resumen_a_tabla(tabla_completa_significativa.round(3), f"Tabla de Correlaciones Significativas (Umbral >= {umbral_heat_map})")

        # === Filtrar datos seleccionados ===
        print("\n=== Filtrando datos seleccionados ===")

        df_filtrado = df_procesado[parametros_seleccionados]

        # Tabla filtrada
        print("\n=== Cálculo de correlaciones filtradas ===")
        tabla_filtrada = df_filtrado.corr()
        agregar_resumen_a_tabla(
            tabla_filtrada.round(3),
            "Tabla de Correlaciones Filtradas por parametros seleccionados (Para Heatmap)",
        )

        # Filtrar correlaciones por el umbral_heat_map para la tabla filtrada
        tabla_filtrada_significativa = tabla_filtrada[
            (tabla_filtrada.abs() >= umbral_heat_map) & (tabla_filtrada != 1)
        ]
        agregar_resumen_a_tabla(
            tabla_filtrada_significativa.round(3),
            f"Tabla de parametros seleccionados, filtrada por Correlaciones Significativas (Umbral >= {umbral_heat_map})",
        )

        # Preparar datos para el heatmap
        print("\n=== Preparando datos para el heatmap ===")
        heatmap_data = df_filtrado.dropna(
            thresh=2
        )  # Excluir variables con menos de 2 valores válidos
        heatmap_correlaciones = heatmap_data.corr()

        # Generar heatmap
        print("\n=== Generando heatmap ===")
        plt.figure(figsize=(12, 10))
        cmap = sns.diverging_palette(10, 145, s=80, l=55, n=9, as_cmap=True)
        sns.heatmap(
            heatmap_correlaciones,
            annot=True,
            cmap=cmap,
            center=0,
            linewidths=0.5,
            vmin=-1,
            vmax=1,
        )
        plt.title(
            f"Heatmap de Correlaciones de Variables Seleccionadas (Umbral >= {umbral_heat_map})"
        )
        plt.show()

    except ValueError as e:
        print(f"Error: {e}. Por favor verifica los parámetros seleccionados.")
        tabla_completa = None  # Asegurar que la variable esté definida
    except KeyError as e:
        print(
            f"Error: {e}. Asegúrate de que las variables seleccionadas existen en los datos."
        )
        tabla_completa = None  # Asegurar que la variable esté definida

    if devolver_tabla:
        return tabla_completa
    else:
        return None


