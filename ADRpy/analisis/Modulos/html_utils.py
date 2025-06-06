import pandas as pd
import numpy as np
from IPython.display import display, HTML



def convertir_a_html(datos_procesados, titulo="", ancho="100%", alto="400px", mostrar=True):
    """
    Convierte un DataFrame o Series a una tabla HTML, redondeando números a 3 cifras significativas.
    :param datos_procesados: DataFrame o Series a transformar.
    :param titulo: Título opcional para mostrar en la tabla.
    :param ancho: Ancho del contenedor HTML.
    :param alto: Alto del contenedor HTML.
    :param mostrar: Si True, muestra la tabla directamente; si False, devuelve el HTML.
    """

    # Asegurarse de que sea un DataFrame
    if isinstance(datos_procesados, pd.Series):
        datos_procesados = datos_procesados.to_frame(name="Valores")
        datos_procesados.index.name = "Índice"

    # Redondear números a 3 cifras significativas sin notación científica
    datos_procesados = datos_procesados.apply(
        lambda col: col.map(
            lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x
        ) if col.dtypes in [np.float64, np.int64] else col
    )

    # Estilo CSS modificado
    estilo_scroll = f"""
    <style>
        .scroll-table {{
            overflow-x: auto;
            overflow-y: auto;
            max-height: {alto};
            max-width: {ancho};
            display: block;
            border: 1px solid #ccc;
            margin-bottom: 20px;
            font-size: 9px;
        }}
        table {{
            border-collapse: collapse;
            width: auto;
            table-layout: auto;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 4px;
            white-space: nowrap;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        td {{
            word-wrap: break-word; /* Permite el ajuste del texto */
            max-width: 150px; /* Ajusta el ancho máximo de las celdas si es necesario */
        }}
    </style>
    """
    # Generar el HTML con el título y la tabla
    tabla_html = estilo_scroll + f"<h3>{titulo}</h3><div class='scroll-table'>{datos_procesados.to_html()}</div>"

    # Mostrar o devolver el HTML
    if mostrar:
        from IPython.display import display, HTML
        display(HTML(tabla_html))  # Muestra directamente
    else:
        return tabla_html  # Devuelve el HTML
