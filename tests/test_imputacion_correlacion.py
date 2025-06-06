import pandas as pd
from ADRpy.analisis.Modulos.imputacion_correlacion import imputacion_correlacion


def test_imputacion_correlacion_basica():
    df, reporte = imputacion_correlacion('ADRpy/analisis/Data/Datos_aeronaves.xlsx')
    assert not df.isna().any().any(), "Deberia imputar todos los valores faltantes"
    # Verificamos que el valor imputado para Potencia en la fila 2 sea cercano al calculo esperado
    valor = df.loc[2, 'Potencia']
    assert round(valor, 3) == round(25.9691788448, 3)
    assert 'Confianza' in reporte.columns
