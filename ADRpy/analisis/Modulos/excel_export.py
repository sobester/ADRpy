import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.comments import Comment




def exportar_excel_con_imputaciones(archivo_origen, df_procesado, resumen_imputaciones, archivo_destino= r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Results\Datos_imputados.xlsx"):
    """
    Exporta el DataFrame procesado a un archivo Excel manteniendo el formato original.
    Agrega colores y comentarios a las celdas imputadas por similitud y correlación.

    :param archivo_origen: Ruta del archivo Excel original.
    :param archivo_destino: Ruta del archivo Excel de salida.
    :param df_procesado: DataFrame con las imputaciones realizadas.
    :param resumen_imputaciones: Lista de diccionarios con detalles de imputaciones.
    """
    try:
        # Asegurarse de que 'resumen_imputaciones' sea una lista de diccionarios
        if isinstance(resumen_imputaciones, pd.DataFrame):
            resumen_imputaciones = resumen_imputaciones.to_dict('records')

        # Manejar el caso donde no haya imputaciones
        if not resumen_imputaciones:
            print("No hay imputaciones para exportar.")
            return

        print(f"=== Exportando datos al archivo: {archivo_destino} ===")
        wb = load_workbook(archivo_origen)
        ws = wb.active

        color_similitud = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        color_correlacion = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")

        imputaciones_por_celda = {
            (registro["Parámetro"], registro["Aeronave"]): registro
            for registro in resumen_imputaciones
        }

        # Recorrer las celdas del archivo original y actualizar según las imputaciones
        for fila in ws.iter_rows(min_row=2, min_col=2):  # Ajustar filas/columnas según tu estructura
            for celda in fila:
                if celda and celda.column and celda.row:  # Validar que la celda y sus propiedades no sean None
                    parametro = ws.cell(row=1, column=celda.column).value  # Obtener nombre del parámetro
                    aeronave = ws.cell(row=celda.row, column=1).value  # Obtener nombre de la aeronave

                    if (parametro, aeronave) in imputaciones_por_celda:
                        registro = imputaciones_por_celda[(parametro, aeronave)]
                        valor_imputado = df_procesado.at[aeronave, parametro]  # Acceder usando filas como aeronaves y columnas como parámetros

                        # Actualizar el valor en la celda
                        celda.value = valor_imputado

                        # Asignar color según el tipo de imputación
                        if registro["Método"] == "Similitud":
                            celda.fill = color_similitud
                        elif registro["Método"] == "Correlación":
                            celda.fill = color_correlacion

                        # Agregar comentario con el nivel de confianza y la iteración
                        comentario = (
                            f"Nivel de confianza: {registro['Nivel de Confianza']*100:.2f}%\n"
                            f"Iteración: {registro['Iteración']}"
                        )
                        celda.comment = Comment(comentario, "Sistema")

        # Guardar el archivo con las imputaciones
        wb.save(archivo_destino)
        print(f"Exportación completada. El archivo se guardó como '{archivo_destino}'.")

    except FileNotFoundError:
        print(f"Error: El archivo '{archivo_origen}' o {archivo_destino} no fue encontrado.")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
