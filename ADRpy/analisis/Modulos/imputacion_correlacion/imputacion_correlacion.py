import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def penalizacion_por_k(k: int) -> float:
    """Return penalization factor based on sample size."""
    if k > 10:
        return 1.0
    return max(
        0,
        min(
            1,
            0.00002281 * k ** 5
            - 0.00024 * k ** 4
            - 0.0036 * k ** 3
            + 0.046 * k ** 2
            + 0.0095 * k
            + 0.024,
        ),
    )


def cargar_y_validar_datos(path: str) -> pd.DataFrame:
    """Load Excel data from the given path using sheet 'data_frame_prueba'."""
    try:
        df = pd.read_excel(path, sheet_name="data_frame_prueba")
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    except ValueError:
        raise ValueError(
            "No se pudo leer la hoja 'data_frame_prueba'. Verifique el archivo"
        )
    df = df.rename(columns=lambda c: str(c).strip())
    df.replace("nan", np.nan, inplace=True)
    return df


def seleccionar_predictores_validos(df: pd.DataFrame, objetivo: str, rango: float = 0.15) -> list:
    """Return numeric predictors with enough data and within ±15% range."""
    numericas = df.select_dtypes(include=[np.number]).columns
    filas_obj = df[df[objetivo].isna()].index
    candidatos = []
    for col in numericas:
        if col == objetivo or df[col].notna().sum() < 5:
            continue
        valido = True
        vals_no_nan = df[col].dropna()
        if vals_no_nan.empty:
            continue
        mn, mx = vals_no_nan.min(), vals_no_nan.max()
        rango_min, rango_max = mn * (1 - rango), mx * (1 + rango)
        for idx in filas_obj:
            val = df.at[idx, col]
            if pd.isna(val) or not (rango_min <= val <= rango_max):
                valido = False
                break
        if valido:
            candidatos.append(col)
    return candidatos


def generar_combinaciones(predictores: list) -> list:
    combos = []
    for r in (1, 2):
        combos.extend(list(combinations(predictores, r)))
    return combos


def entrenar_modelo(
    df: pd.DataFrame, objetivo: str, predictores: tuple, poly: bool
) -> dict | None:
    """Train linear or polynomial model and compute metrics."""
    df_train = df.dropna(subset=[objetivo, *predictores])
    if len(df_train) < len(predictores) + 1:
        return None
    X = df_train[list(predictores)]
    y = df_train[objetivo]
    if poly:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        X_trans = pf.fit_transform(X)
    else:
        pf = None
        X_trans = X
    modelo = LinearRegression().fit(X_trans, y)
    pred = modelo.predict(X_trans)
    mape = mean_absolute_percentage_error(y, pred) * 100
    r2 = r2_score(y, pred)
    corr = 0.6 * (r2 / 0.7) + 0.4 * (1 - mape / 15)
    confianza = corr * penalizacion_por_k(len(df_train))
    tipo = ("poly" if poly else "linear") + f"-{len(predictores)}"
    return {
        "predictores": predictores,
        "modelo": modelo,
        "pf": pf,
        "mape": mape,
        "r2": r2,
        "corr": corr,
        "confianza": confianza,
        "tipo": tipo,
        "n": len(df_train),
    }


def filtrar_mejores_modelos(modelos: list, top: int = 2) -> list:
    """Return top models per type based on confianza."""
    modelos = [m for m in modelos if m is not None and m["mape"] <= 15 and m["r2"] >= 0.7]
    grupos: defaultdict[str, list] = defaultdict(list)
    for m in modelos:
        grupos[m["tipo"]].append(m)
    mejores = []
    for lst in grupos.values():
        lst.sort(key=lambda x: x["confianza"], reverse=True)
        mejores.extend(lst[:top])
    return mejores


def validar_con_loocv(df: pd.DataFrame, objetivo: str, info: dict) -> tuple:
    """Return MAE and R2 from LOOCV validation."""
    df_train = df.dropna(subset=[objetivo, *info["predictores"]])
    if df_train.empty:
        return np.inf, -np.inf
    X = df_train[list(info["predictores"])]
    y = df_train[objetivo]
    if info["pf"] is not None:
        X = info["pf"].fit_transform(X)
        X_vals = X
    else:
        X_vals = X.values
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for i, (train_idx, test_idx) in enumerate(loo.split(X_vals)):
        m = LinearRegression().fit(X_vals[train_idx], y.iloc[train_idx])
        preds[i] = m.predict(X_vals[test_idx])[0]
    mae = float(np.mean(np.abs(y - preds)))
    r2 = float(r2_score(y, preds))
    return mae, r2


def imputar_valores(df: pd.DataFrame, objetivo: str, info: dict):
    df_res = df.copy()
    faltantes = df_res[df_res[objetivo].isna()].index
    imputaciones = []
    if not len(faltantes):
        return df_res, imputaciones
    X_pred = df_res.loc[faltantes, list(info["predictores"])]
    if info["pf"] is not None:
        X_pred = info["pf"].transform(X_pred)
    valores = info["modelo"].predict(X_pred)
    df_res.loc[faltantes, objetivo] = valores
    vals_no_nan = df[list(info["predictores"])]
    advert = ""
    mn = vals_no_nan.min()
    mx = vals_no_nan.max()
    for idx, val in zip(faltantes, valores):
        advert = ""
        for col in info["predictores"]:
            v = df.at[idx, col]
            if pd.isna(v) or not (mn[col] * 0.85 <= v <= mx[col] * 1.15):
                advert = "Extrapolacion"
                break
        imputaciones.append(
            {
                "Fila": idx,
                "Parametro": objetivo,
                "Valor imputado": val,
                "Confianza": info["confianza"],
                "Corr": info["corr"],
                "k": info["n"],
                "Tipo Modelo": info["tipo"],
                "Predictores": ",".join(info["predictores"]),
                "Penalizacion_k": penalizacion_por_k(info["n"]),
                "Advertencia": advert,
            }
        )
    return df_res, imputaciones


def generar_reporte_final(registros: list) -> pd.DataFrame:
    return pd.DataFrame(registros)


def imputacion_correlacion(path: str = "ADRpy/analisis/Data/Datos_aeronaves.xlsx"):
    df = cargar_y_validar_datos(path)
    reporte = []
    for objetivo in [c for c in df.columns if df[c].isna().any()]:
        predictores = seleccionar_predictores_validos(df, objetivo)
        if not predictores:
            continue
        modelos = []
        for combo in generar_combinaciones(predictores):
            for poly in (False, True):
                modelos.append(entrenar_modelo(df, objetivo, combo, poly))
        mejores = filtrar_mejores_modelos(modelos)
        if not mejores:
            continue
        mejor = min(
            ((m, validar_con_loocv(df, objetivo, m)) for m in mejores),
            key=lambda t: t[1][0],
        )[0]
        df, imps = imputar_valores(df, objetivo, mejor)
        reporte.extend(imps)
    return df, generar_reporte_final(reporte)
