# 📊 Diagrama de flujo de datos - Proyecto ADRpy

Este diagrama representa el flujo principal de los datos en el proceso de análisis e imputación de aeronaves.


```mermaid
graph TD
    A[Archivo Excel original] --> B[Procesamiento y limpieza de datos]
    B --> C[Analisis de correlaciones]
    B --> D[Imputacion por similitud MTOW]
    C --> E[Imputacion por correlacion]
    D --> F[Bucle de imputacion hasta completar valores]
    E --> F
    F --> G[Exportacion a nuevo Excel con formato y comentarios]
    G --> H[Visualizacion con Data Wrangler o HTML]
```

```mermaid
graph TD
    A[📁 Excel original] --> B[🧹 Limpieza de datos]
    B --> C[📈 Correlaciones]
    B --> D[🧮 Similitud MTOW]
    C --> E[🔗 Correlaciones significativas]
    D --> F[🔁 Bucle de imputacion]
    E --> F
    F --> G[📤 Exportar a Excel]
    G --> H[🖥️ Visualizar con Data Wrangler]
```
```mermaid
graph TD
    classDef inicio fill=#cce5ff,stroke=#004085,color=#004085,stroke-width:2px;
    classDef proceso fill=#e2e3e5,stroke=#383d41,color=#383d41;
    classDef importante fill=#fff3cd,stroke=#856404,color=#856404,stroke-width:2px;
    classDef salida fill=#d4edda,stroke=#155724,color=#155724;

    A[📁 Excel original]:::inicio --> B[🧹 Limpieza de datos]:::proceso
    B --> C[📈 Correlaciones]:::proceso
    B --> D[🧮 Similitud MTOW]:::proceso
    C --> E[🔗 Correlación significativa]:::importante
    D --> F[🔁 Bucle de imputación]:::importante
    E --> F
    F --> G[📤 Exportación Excel]:::salida
    G --> H[🖥️ Visualización HTML o Data Wrangler]:::salida

---

## 📌 Notas:
- Cada bloque representa una etapa clave del proyecto.
- Las flechas indican la evolución del DataFrame a través de funciones.
- Las funciones clave están agrupadas en módulos dentro de la carpeta `Modulos`.

