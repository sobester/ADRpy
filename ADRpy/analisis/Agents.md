# Archivo agents.md para Codex - Script ADRpy de Imputación por Correlación

## Contexto del proyecto

El script pertenece al proyecto ADRpy, enfocado en la imputación de datos faltantes en un DataFrame con información técnica aeronáutica. El objetivo específico es imputar valores faltantes utilizando modelos predictivos basados en correlaciones estadísticas.

## Objetivo del script

* Automatizar la imputación de valores faltantes mediante regresión lineal y polinómica (grado 2).
* Evaluar todas las combinaciones posibles de hasta 2 predictores, considerando métricas como MAPE, R², coeficiente Corr, y Confianza.
* Seleccionar automáticamente el mejor modelo para imputar, priorizando la precisión, robustez estadística y capacidad de generalización (evaluada con LOOCV).

## Entradas necesarias

* importar excel con nombre datos aeronaves dentro del mismo directorio y que use la pestaña que se llama (data\_frame\_prueba)
* manejar errores comunes y como encabezados con caracteres o espacios raros, y detectar las celdas vacias manejando diferentes opciones en excel las celdas vacias estan escritas con "nan"

## Salidas esperadas

* DataFrame actualizado con valores imputados.
* Reporte detallado con:

  * Modelo usado (ecuación final).
  * Métricas: MAPE, R², Corr, Confianza y Correlación final.
  * Advertencias explícitas en caso de extrapolaciones (valores fuera del rango de entrenamiento).

## Flujo lógico principal detallado

### Paso 1: Detección de celda objetivo

* Identificar la celda faltante a imputar.

### Paso 2: Filtrado por familia

* Filtrar por tipo de misión de la aeronave objetivo.
* Si este filtro deja menos de 5 muestras completas para entrenar, relajar el filtro y usar dataset completo, registrando advertencia explícita.

### Paso 3: Selección inicial de predictores

* En esta etapa realizamos dos verificaciones, por un lugar garantizamos la existencia de valores en la aeronave objetivo para utilizarlos como valor de entrada en la ecuación de regresión que va a resultar de entrenar el modelo.
* En segundo lugar, validamos que los valores de estos parámetros se encuentren dentro del dominio de entrenamiento del modelo (rango ±15%) de valores del parámetro predictor. o sea cada parametro disponible en la aeronave se debe chequear que se encuentre dentro del rango aceptable dentro de ese mismo parametro comparandolo con los valores minimos y maximos de ese parametro sin ser la aeronave que se está evaluando.

### Paso 4: Generación de combinaciones

* Crear todas las combinaciones posibles de hasta 2 predictores disponibles.

### Paso 5: Evaluación de cantidad de datos

* Para cada combinación, filtrar y asegurar la existencia mínima de datos válidos necesarios según la cantidad de coeficientes del modelo.

### Paso 6: Entrenamiento del modelo

* Entrenar modelos (lineales y polinómicos de grado 2).
* Aplicar normalización en caso de regresión polinómica.
* Calcular métricas iniciales para filtrar modelos:

  * Coeficiente Corr:
    $Corr = 0.6 \times \frac{R^2}{0.7} + 0.4 \times \left(1 - \frac{MAPE}{15}\right)$
  * Confianza final (tras aplicar Corr)
* Manejar errores posibles por matrices indefinidas brindando las advertencias correspondientes.

### Paso 7: Filtrado y selección de los mejores modelos

* Filtrar modelos según MAPE y R² mínimos establecidos.
* Seleccionar los mejores 2 modelos de cada tipo basándose en Confianza final.

### Paso 8: Validación cruzada (LOOCV)

* Aplicar Leave-One-Out Cross-Validation para evaluar capacidad de generalización de los modelos seleccionados.
* Seleccionar el modelo con mejor desempeño generalizado (menor error en LOOCV).

### Paso 9: Imputación final

* Imputar el valor faltante utilizando el mejor modelo seleccionado.
* Registrar advertencias en caso de extrapolación fuera del dominio entrenado.

### Paso 10: Generación del reporte

* Crear un DataFrame con un resumen detallado de imputaciones realizadas.
* Registrar métricas detalladas:&#x20;

  * El valor imputado
  * El valor de confianza (calculado tras validación)
  * El número de iteración (en verdad esto se agrega en la función del Loop)
  * El coeficiente Corr del modelo original
  * El número de observaciones k usado para la función
  * El tipo de modelo utilizado (lineal / polinómico)
  * La ecuación del modelo entrenado
  * La cantidad y nombres de predictores involucrados
  * El factor de penalización aplicado (k)
  * Cualquier advertencia relevante (ej. extrapolación, uso de datos sin filtrar, validación inestable)
  * El estado de validez del modelo final (válido / parcialmente válido)
  * Comparación entre coeficientes MAE y R2 de modelo entrenado con todos los datos vs. validación LOOCV   

## Consideraciones técnicas importantes

* Aplicar normalización a predictores al entrenar modelos polinómicos.
* Calcular número de condición para evaluar estabilidad numérica.
* Penalización en la confianza final por baja cantidad de datos: corr x f(k) (con función empírica definida según cantidad de datos disponibles).
* f (k) **=** 0.00002281 **\*** (k/**2)**\*\***5 **************************************************************************************************************************************************************************************************************-************************************************************************************************************************************************************************************************************** 0.00024 **************************************************************************************************************************************************************************************************************\*************************************************************************************************************************************************************************************************************** (k**/**2)**\*\***4 **************************************************************************************************************************************************************************************************************-************************************************************************************************************************************************************************************************************** 0.0036 **************************************************************************************************************************************************************************************************************\*************************************************************************************************************************************************************************************************************** (k**/**2)**\*\***3 **************************************************************************************************************************************************************************************************************+************************************************************************************************************************************************************************************************************** 0.046 **************************************************************************************************************************************************************************************************************\*************************************************************************************************************************************************************************************************************** (k**/**2)**\*\***2 **************************************************************************************************************************************************************************************************************+************************************************************************************************************************************************************************************************************** 0.0095 **************************************************************************************************************************************************************************************************************\*************************************************************************************************************************************************************************************************************** (k**/\*\*2) **+** 0.024
* Validar dominio estricto: valores fuera del rango entrenado (±15%) deben generar advertencias explícitas.

## Estructura del código sugerida

dentro de una capeta que sea especificamente (imputación\_correlacion) hacer la modularización de la función adrpy/analisis/modulos/(aqui carpeta)/aqui modulos

* Claridad modular con funciones específicas:

  * `cargar_y_validar_datos()`
  * `filtrar_por_familia()`
  * `seleccionar_predictores_validos()`
  * `generar_combinaciones()`
  * `entrenar_modelos_y_evaluar()`
  * `filtrar_mejores_modelos()`
  * `validar_con_loocv()`
  * `imputar_valores()`
  * `generar_reporte_final()`

## Paquetes necesarios

* `pandas`, `numpy`, `scikit-learn`, `matplotlib` (opcional).
