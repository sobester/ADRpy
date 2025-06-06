import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from Modulos.html_utils import convertir_a_html


def procesar_datos_y_manejar_duplicados(df, respuesta_global=None):
    """
    Limpia un DataFrame preservando la estructura original y maneja duplicados en índices y columnas.
    Incluye interacción para gestionar duplicados según las elecciones del usuario.
    :param df: DataFrame a procesar.
    :return: DataFrame limpio y procesado.
    """
    try:
        print("=== Inicio del procesamiento de datos ===")

        # Paso 1: Limpieza inicial de encabezados
        df.columns = df.columns.str.strip().str.replace("\xa0", " ", regex=True)
        df.index = df.index.astype(str).str.strip().str.replace("\xa0", " ", regex=True)

        # Paso 2: Eliminar filas y columnas completamente vacías
        df.dropna(how="all", inplace=True)  # Filas vacías
        df.dropna(how="all", axis=1, inplace=True)  # Columnas vacías

        # Paso 3: Manejo de duplicados
        print("\n=== Comprobación de duplicados ===")
        duplicados_fila = df.index[df.index.duplicated()].tolist()
        duplicados_columna = df.columns[df.columns.duplicated()].tolist()

        if not duplicados_fila and not duplicados_columna:
            print("No se encontraron duplicados en índices o columnas.")
        else:
            print(f"Índices duplicados: {duplicados_fila}")
            print(f"Columnas duplicadas: {duplicados_columna}")

            # Crear ventana emergente para interacción
            if respuesta_global is None:
                root = tk.Tk()
                root.withdraw()
                respuesta_global = simpledialog.askstring(
                    "Manejo global de duplicados",
                    "Se encontraron duplicados. ¿Deseas aplicar una acción global a todos?\n"
                    "[1] Eliminar todos los duplicados\n"
                    "[2] Conservar el primero en todos\n"
                    "[3] Conservar el último en todos\n"
                    "[4] Procesar duplicados uno por uno",
                )

            # Aplicar acción global si corresponde
            if respuesta_global in ["1", "2", "3"]:
                if respuesta_global == "1":
                    print("Eliminando todos los duplicados...")
                    if duplicados_fila:
                        df = df.loc[~df.index.duplicated(keep=False)]
                    if duplicados_columna:
                        df = df.loc[:, ~df.columns.duplicated(keep=False)]

                elif respuesta_global == "2":
                    print("Conservando el primero en todos los duplicados...")
                    if duplicados_fila:
                        df = df.loc[~df.index.duplicated(keep="first")]
                    if duplicados_columna:
                        df = df.loc[:, ~df.columns.duplicated(keep="first")]

                elif respuesta_global == "3":
                    print("Conservando el último en todos los duplicados...")
                    if duplicados_fila:
                        df = df.loc[~df.index.duplicated(keep="last")]
                    if duplicados_columna:
                        df = df.loc[:, ~df.columns.duplicated(keep="last")]
            else:
                # Procesar duplicados uno por uno si respuesta_global es '4'
                for duplicado in duplicados_fila + duplicados_columna:
                    tipo = "Índice" if duplicado in duplicados_fila else "Columna"
                    respuesta = simpledialog.askstring(
                        "Duplicado encontrado",
                        f"{tipo} duplicado '{duplicado}' encontrado. Opciones:\n"
                        "[1] Eliminar\n"
                        "[2] Conservar el primero\n"
                        "[3] Conservar el último",
                    )
                    # Realizar la acción según la elección del usuario
                    if respuesta == "1":
                        if tipo == "Índice":
                            df = df[df.index != duplicado]
                        else:
                            df = df.loc[:, df.columns != duplicado]
                    elif respuesta == "2":
                        if tipo == "Índice":
                            df = df.loc[~df.index.duplicated(keep="first")]
                        else:
                            df = df.loc[:, ~df.columns.duplicated(keep="first")]
                    elif respuesta == "3":
                        if tipo == "Índice":
                            df = df.loc[~df.index.duplicated(keep="last")]
                        else:
                            df = df.loc[:, ~df.columns.duplicated(keep="last")]

        # Paso 4: Convertir valores internos a numéricos
        print("\n=== Convirtiendo valores a numéricos donde sea posible ===")
        for col in df.columns:
            try:
                df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as e:
                print(
                    f"Advertencia: No se pudo convertir la columna '{col}' a numérico. Error: {e}"
                )

        print("=== Procesamiento completado ===")
        return df

    except Exception as e:
        raise ValueError(f"Error durante el procesamiento y manejo de duplicados: {e}")


def mostrar_celdas_faltantes_con_seleccion(df, fila_seleccionada=None, debug_mode=False):
    """
    Muestra las celdas faltantes de una fila específica elegida por el usuario o automáticamente.

    :param df: DataFrame a analizar.
    :param fila_seleccionada: Nombre de la fila a analizar. Si None, se pedirá al usuario o se usará el modo automático.
    :param debug_mode: Si True, selecciona automáticamente la primera fila con datos faltantes si no se pasa ninguna.
    :return: DataFrame con las celdas faltantes de la fila seleccionada.
    """
    aeronaves_con_nulos = df.index[df.isnull().any(axis=1)].tolist()

    if not aeronaves_con_nulos:
        print("✅ No hay filas con valores faltantes.")
        return pd.DataFrame()

    if debug_mode and not fila_seleccionada:
        fila_seleccionada = aeronaves_con_nulos[0]
        print(f"[DEBUG] Seleccionando automáticamente la primera fila con nulos: '{fila_seleccionada}'")

    elif not fila_seleccionada:
        print("\n=== Filas con datos faltantes ===")
        for i, fila in enumerate(aeronaves_con_nulos, start=1):
            print(f"{i}. {fila}")

        seleccion = input("Selecciona el número de la fila a analizar (presiona Enter para seleccionar la primera): ").strip()

        if not seleccion.isdigit():
            print("🔁 Entrada inválida o vacía. Seleccionando la primera fila por defecto.")
            fila_seleccionada = aeronaves_con_nulos[0]
        else:
            seleccion = int(seleccion) - 1
            if 0 <= seleccion < len(aeronaves_con_nulos):
                fila_seleccionada = aeronaves_con_nulos[seleccion]
            else:
                print("🔁 Número fuera de rango. Seleccionando la primera fila por defecto.")
                fila_seleccionada = aeronaves_con_nulos[0]

    print(f"\n=== Analizando celdas faltantes en la fila: '{fila_seleccionada}' ===")
    celdas_faltantes = df.loc[fila_seleccionada][df.loc[fila_seleccionada].isnull()]

    return celdas_faltantes


def generar_resumen_faltantes(
    df, titulo="Resumen de Valores Faltantes por Fila", ancho="50%", alto="300px"
):
    """
    Genera un resumen de los valores faltantes por fila en un DataFrame.
    También genera una tabla HTML con la sumatoria total de los valores faltantes de todas las filas.

    :param df: DataFrame a analizar.
    :param titulo: Título opcional para mostrar en la tabla HTML.
    :param ancho: Ancho del contenedor HTML.
    :param alto: Alto del contenedor HTML.
    :return: Tuple con dos DataFrames: resumen de valores faltantes por fila y sumatoria total.
    """
    # Calcular la cantidad de valores faltantes por fila
    faltantes_por_fila = df.isnull().sum(axis=1)

    # Crear un DataFrame con el resumen por fila
    resumen_faltantes = faltantes_por_fila.reset_index()
    resumen_faltantes.columns = ["Fila", "Valores Faltantes"]

    # Calcular la sumatoria total de los valores faltantes
    total_faltantes = faltantes_por_fila.sum()
    resumen_total = pd.DataFrame(
        {"Resumen": ["Total de Valores Faltantes"], "Cantidad": [total_faltantes]}
    )

    # Mostrar el resumen por fila como una tabla HTML
    convertir_a_html(
        resumen_faltantes, titulo=titulo, ancho=ancho, alto=alto, mostrar=True
    )

    # Mostrar la sumatoria total como una tabla HTML
    convertir_a_html(
        resumen_total,
        titulo="Sumatoria Total de Valores Faltantes",
        ancho=ancho,
        alto="100px",
        mostrar=True,
    )

    # Retornar ambos DataFrames para su posible uso posterior
    return resumen_faltantes, resumen_total
