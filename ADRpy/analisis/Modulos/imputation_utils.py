import pandas as pd
from .html_utils import convertir_a_html

def imprimir_detalles_imputacion(numero_valor_imputado, parametro, aeronave, mtow_actual, rango_min, rango_max, candidatas_validas, detalles_ajustes, valores_ajustados, valor_imputado, confianza, calculos_confianza):
    """
    Imprime un resumen claro y organizado del proceso de imputación en consola.
    """
    negrita = "\033[1m"
    reset = "\033[0m"

    print(f"\n{negrita}======================== DETALLE DE CÁLCULO DE IMPUTACIÓN #{numero_valor_imputado} ========================{reset}")
    print(f"{negrita}Parámetro:{reset} {parametro}")
    print(f"{negrita}Aeronave a imputar:{reset} {aeronave}")
    print(f"{negrita}MTOW actual:{reset} {mtow_actual} kg")
    print(f"{negrita}Rango Similitud:{reset} {rango_min*100:.0f}% - {rango_max*100:.0f}%")
    print(f"{negrita}Candidatas dentro del rango:{reset} {', '.join(candidatas_validas.index)}")

    print("\nAeronaves Válidas para el Cálculo:")
    print("-----------------------------------------------------------------------------------------------")
    print("Aeronave    | MTOW Candidata | Rel. MTOW (Candidata/Actual) | Ajuste Individual | Valor Original | Valor Ajustado")
    print("------------|----------------|------------------------------|-------------------|----------------|---------------")
    for detalle in detalles_ajustes:
        print(f"{detalle['Aeronave']:12}| {detalle['MTOW Candidata']:14}| {detalle['Relación MTOW']:<30.3f}| {detalle['Ajuste Individual']:<19.4f}| {detalle['Valor Original']:<16.2f}| {detalle['Valor Ajustado']:<13.2f}")
    print("-----------------------------------------------------------------------------------------------")

    print("\nCálculo del Valor Final:")
    print(f"{negrita}- Se tomó la mediana de los valores ajustados {valores_ajustados} = {valor_imputado:.2f}{reset}")
    print(f"{negrita}- Nivel de Confianza calculado:{reset} {confianza:.2f}")
    print(f"{negrita}- Valor Imputado Final:{reset} {valor_imputado:.2f}")

    print("\nDetalle del Cálculo de Confianza:")
    for key, value in calculos_confianza.items():
        print(f"{negrita}- {key}:{reset} {value:.2f}")
    print(f"{negrita}============================================================================================{reset}\n")
