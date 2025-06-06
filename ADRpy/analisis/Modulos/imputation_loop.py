import pandas as pd
from Modulos.imputacion_similitud_flexible import *
from .correlation_imputation import Imputacion_por_correlacion
from .html_utils import convertir_a_html
from .data_processing import generar_resumen_faltantes

def bucle_imputacion_similitud_correlacion(
    df_parametros,
    df_atributos,
    parametros_preseleccionados,
    bloques_rasgos,
    capas_familia,
    df_procesado,
    df_filtrado,
    tabla_completa,
    parametros_seleccionados,
    umbral_correlacion=0.7,
    rango_min=0.85,
    rango_max=1.15,
    nivel_confianza_min_similitud=0.7,
    max_iteraciones=10,
    reduccion_confianza=0.05,
    nivel_confianza_min_correlacion=None,
    debug_mode=False
):

    """
    Realiza un bucle alternando imputaciones por similitud y correlación, consolidando los resultados.
    Ahora se evita actualizar los DataFrames inmediatamente, y se eligen las imputaciones finales
    al final de cada iteración.

    Retorna:
        df_procesado_base (pd.DataFrame): DataFrame con imputaciones realizadas.
        df_resumen (pd.DataFrame): Detalle consolidado de imputaciones realizadas.
    """

    df_procesado_base = df_procesado.copy()  # Copia base del DataFrame original
    df_filtrado_base = df_filtrado.copy()  # Copia base del DataFrame original

    convertir_a_html(
        df_procesado_base,
        titulo="df_procesado_base",
        ancho="100%",
        alto="400px",
        mostrar=True,
    )
    convertir_a_html(
        df_filtrado_base,
        titulo="df_filtrado_base",
        ancho="100%",
        alto="400px",
        mostrar=True,
    )
    resumen_imputaciones = (
        []
    )  # Lista para consolidar detalles de todas las imputaciones finales

    # Configuración inicial para imputaciones por similitud
    print("\n=== Configuración Inicial ===")
    
    from Modulos.user_interaction import (
        solicitar_rango_min,
        solicitar_rango_max,
        solicitar_confianza_min_similitud,
    )

    rango_min = rango_min if debug_mode else solicitar_rango_min()
    rango_max = rango_max if debug_mode else solicitar_rango_max()
    nivel_confianza_min_similitud = nivel_confianza_min_similitud if debug_mode else solicitar_confianza_min_similitud()


    print(
        f"\nValores configurados: Rango MTOW [{rango_min*100:.0f}% - {rango_max*100:.0f}%], Confianza Mínima: {nivel_confianza_min_similitud:.2f}"
    )

    # Configuración inicial para imputaciones por correlación
    from Modulos.user_interaction import (
        solicitar_umbral_correlacion,
        solicitar_confianza_min_correlacion,
    )

    umbral_correlacion = (
        umbral_correlacion if debug_mode and umbral_correlacion is not None else solicitar_umbral_correlacion()
    )
    nivel_confianza_min_correlacion = (
        nivel_confianza_min_correlacion if debug_mode and nivel_confianza_min_correlacion is not None
        else solicitar_confianza_min_correlacion()
    )

    # Definir valores predeterminados para correlación
    min_datos_validos = 5  # Cantidad mínima de datos requeridos por parámetro
    max_lineas_consola = 40000000

    for iteracion in range(1, max_iteraciones + 1):
        print("\n" + "=" * 80)
        print(f"\033[1m=== INICIO DE ITERACIÓN {iteracion} ===\033[0m")
        print("=" * 80)

        print(f"\n=== Iteración {iteracion}: Resumen antes de imputaciones ===")
        resumen_antes, total_faltantes_antes = generar_resumen_faltantes(
            df_procesado_base,
            titulo=f"Resumen de Valores Faltantes Antes de Iteración {iteracion}",
        )

        # Crear copias independientes para cada método
        df_similitud = df_procesado_base.copy()
        df_correlacion = df_procesado_base.copy()

        # Imputación por similitud (no actualiza todavía)
        print("\n" + "-" * 80)
        print(f"\033[1m*** IMPUTACIÓN POR SIMILITUD - ITERACIÓN {iteracion} ***\033[0m")
        print("-" * 80)
        df_similitud_resultado = df_similitud.copy()
        reporte_similitud = []

        for parametro in parametros_preseleccionados:
            for aeronave in df_similitud_resultado.index:  # Acceder usando filas como aeronaves y columnas como parámetros
                if pd.isna(df_similitud_resultado.at[aeronave, parametro]):
                    resultado = imputar_por_similitud(
                        df_parametros=df_parametros,
                        df_atributos=df_atributos,
                        aeronave_obj=aeronave,
                        parametro_objetivo=parametro,
                        bloques_rasgos=bloques_rasgos,
                        capas_familia=capas_familia
                    )

                    if resultado is not None:
                        df_similitud_resultado.at[aeronave, parametro] = resultado["valor"]  # Corregir lógica para asignar valores
                        reporte_similitud.append({
                            "Aeronave": aeronave,
                            "Parámetro": parametro,
                            "Valor Imputado": resultado["valor"],
                            "Nivel de Confianza": resultado["confianza"]
                        })

        if reporte_similitud and len(reporte_similitud) > 0:
            print("\033[1m>>> RESULTADOS DE IMPUTACIÓN POR SIMILITUD\033[0m")
            # Se guardan las imputaciones de similitud, pero NO se actualiza el DataFrame aún.
            # Se agregan la iteración y método aquí.
            for registro in reporte_similitud:
                registro["Iteración"] = iteracion
                registro["Método"] = "Similitud"
                registro["Nivel de Confianza"] *= (1 - reduccion_confianza) ** (
                    iteracion - 1
                )
        else:
            print(
                "\033[1mNo se realizaron imputaciones por similitud en esta iteración.\033[0m"
            )

        # Imputación por correlación (no actualiza todavía)
        print("\n" + "-" * 80)
        print(
            f"\033[1m*** IMPUTACIÓN POR CORRELACIÓN - ITERACIÓN {iteracion} ***\033[0m"
        )
        print("-" * 80)
        df_correlacion_final, reporte_correlacion = Imputacion_por_correlacion(
            df_correlacion,
            parametros_preseleccionados,
            tabla_completa,
            parametros_seleccionados,
            min_datos_validos=min_datos_validos,
            max_lineas_consola=max_lineas_consola,
            umbral_correlacion=umbral_correlacion,
            nivel_confianza_min_correlacion=nivel_confianza_min_correlacion,            
        )

        if reporte_correlacion and len(reporte_correlacion) > 0:
            print("\033[1m>>> RESULTADOS DE IMPUTACIÓN POR CORRELACIÓN\033[0m")
            for registro in reporte_correlacion:
                registro["Iteración"] = iteracion
                registro["Método"] = "Correlación"
                registro["Nivel de Confianza"] *= (1 - reduccion_confianza) ** (
                    iteracion - 1
                )
        else:
            print(
                "\033[1mNo se realizaron imputaciones por correlación en esta iteración.\033[0m"
            )

        # Combinar las imputaciones de similitud y correlación
        imputaciones_candidatas = {}

        def registrar_imputacion(regs):
            for reg in regs:
                parametro = reg["Parámetro"]
                aeronave = reg["Aeronave"]
                key = (parametro, aeronave)
                if key not in imputaciones_candidatas:
                    imputaciones_candidatas[key] = []
                imputaciones_candidatas[key].append(reg)

        if reporte_similitud and len(reporte_similitud) > 0:
            registrar_imputacion(reporte_similitud)
        if reporte_correlacion and len(reporte_correlacion) > 0:
            registrar_imputacion(reporte_correlacion)

        # Seleccionar las mejores imputaciones por celda (la de mayor confianza)
        imputaciones_finales = []
        for key, candidatos in imputaciones_candidatas.items():
            parametro, aeronave = key
            # Si ya hay un valor en df_procesado_base, no imputar.
            if not pd.isna(df_procesado_base.at[aeronave, parametro]):  # Cambiar lógica para trabajar con filas como aeronaves y columnas como parámetros
                continue
            # Escoger la de mayor confianza
            mejor = max(candidatos, key=lambda x: x["Nivel de Confianza"])
            imputaciones_finales.append(mejor)

        # Ahora sí, aplicar las imputaciones finales al DataFrame base
        for imp in imputaciones_finales:
            parametro = imp["Parámetro"]
            aeronave = imp["Aeronave"]
            valor = imp["Valor Imputado"]
            metodo = imp["Método"]
            df_procesado_base.at[aeronave, parametro] = valor  # Corregir lógica para asignar valores
            df_filtrado_base.at[aeronave, parametro] = valor  # Corregir lógica para asignar valores
            resumen_imputaciones.append(imp)
            print(
                f"Imputación final aplicada: {parametro} - {aeronave} = {valor} ({metodo})"
            )

        print(f"\n=== Iteración {iteracion}: Resumen después de imputaciones ===")
        resumen_despues, total_faltantes_despues = generar_resumen_faltantes(
            df_filtrado_base,
            titulo=f"Resumen de Valores Faltantes Después de Iteración {iteracion}",
        )

        # Verificar condición de salida
        no_similitud = reporte_similitud is None or len(reporte_similitud) == 0
        no_correlacion = reporte_correlacion is None or len(reporte_correlacion) == 0
        if no_similitud and no_correlacion:
            print("\033[1mNo se realizaron nuevas imputaciones. Finalizando...\033[0m")
            # Retornar resultados actuales antes de salir
            return df_procesado_base, pd.DataFrame(resumen_imputaciones)

        print("\n" + "=" * 80)
        print(f"\033[1m=== FIN DE ITERACIÓN {iteracion} ===\033[0m")
        print("=" * 80)

    # Si se terminan las iteraciones sin break:
    df_resumen = pd.DataFrame(resumen_imputaciones)
    print("\n" + "=" * 80)
    print("\033[1m=== RESUMEN FINAL ===\033[0m")
    print("=" * 80)

    convertir_a_html(
        df_procesado_base, titulo="DataFrame Procesado Final (df_procesado_base)"
    )
    convertir_a_html(df_resumen, titulo="Resumen Final de Imputaciones (df_resumen)")

    print(f"\033[1mTotal de iteraciones realizadas: {iteracion}\033[0m")
    print(f"\033[1mTotal de valores imputados: {len(resumen_imputaciones)}\033[0m")

    return df_procesado_base, df_resumen
