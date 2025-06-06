def seleccionar_parametros_por_indices(parametros_disponibles, parametros_preseleccionados, entrada_indices=None):
    print("\n=== Selección de Parámetros ===")
    print("Parámetros disponibles:")
    for i, parametro in enumerate(parametros_disponibles, 1):
        print(f"{i}. {parametro}")

    print(f"\nPreseleccionados:  {', '.join(str(parametros_disponibles.index(p) + 1) for p in parametros_preseleccionados if p in parametros_disponibles)}")

    if entrada_indices is None:
        indices = input("Ingresa los números separados por coma (o presiona Enter para usar los preseleccionados): ")
    else:
        indices = entrada_indices

    if not indices.strip():
        seleccionados = parametros_preseleccionados
    else:
        try:
            indices = [int(i.strip()) - 1 for i in indices.split(",")]
            seleccionados = [parametros_disponibles[i] for i in indices]
        except Exception as e:
            print(f"Error al interpretar los índices: {e}")
            seleccionados = parametros_preseleccionados

    print("Parámetros seleccionados después de filtrar:")
    print(seleccionados)
    return seleccionados


def solicitar_umbral(valor_por_defecto=0.7, umbral_manual=None):
    """
    Solicita al usuario ingresar un umbral para las correlaciones significativas.
    Si el usuario no proporciona un valor válido, se usa el valor por defecto.
    :param valor_por_defecto: Valor predeterminado del umbral si el usuario no ingresa ninguno.
    :return: Umbral de correlación como flotante.
    """
    if umbral_manual is not None:
        if not (0 < umbral_manual < 1):
            print(f"⚠️ El umbral_manual ({umbral_manual}) está fuera de rango. Se usará el valor por defecto ({valor_por_defecto})")
            return valor_por_defecto
        return umbral_manual

    try:
        umbral = float(input(f"Ingrese el umbral mínimo de correlación significativa (por defecto {valor_por_defecto}): ") or valor_por_defecto)
        if not (0 < umbral < 1):
            raise ValueError("El umbral debe estar entre 0 y 1.")
        return umbral
    except ValueError as e:
        print(f"Valor inválido: {e}. Se usará el umbral por defecto de {valor_por_defecto}.")
        return valor_por_defecto
    
def solicitar_rango_min(valor_por_defecto=0.85, valor=None):
    try:
        if valor is None:
            valor = input(f"Ingrese el rango mínimo de MTOW (0-2, predeterminado {valor_por_defecto * 100:.0f}): ")
            valor = float(valor) / 100 if valor else valor_por_defecto
        if not (0.01 <= valor <= 2.00):
            raise ValueError
        return valor
    except ValueError:
        print("Valor inválido. Usando el valor predeterminado.")
        return valor_por_defecto


def solicitar_rango_max(valor_por_defecto=1.15, valor=None):
    try:
        if valor is None:
            valor = input(f"Ingrese el rango máximo de MTOW (0-2, predeterminado {valor_por_defecto * 100:.0f}): ")
            valor = float(valor) / 100 if valor else valor_por_defecto
        if not (0.01 <= valor <= 2.00):
            raise ValueError
        return valor
    except ValueError:
        print("Valor inválido. Usando el valor predeterminado.")
        return valor_por_defecto


def solicitar_confianza_min_similitud(valor_por_defecto=0.5, valor=None):
    try:
        if valor is None:
            valor = input(f"Ingrese el nivel mínimo de confianza (0-1, predeterminado {valor_por_defecto}): ")
            valor = float(valor) if valor else valor_por_defecto
        if not (0 <= valor <= 1):
            raise ValueError
        return valor
    except ValueError:
        print("Valor inválido. Usando el valor predeterminado.")
        return valor_por_defecto

def solicitar_umbral_correlacion(valor_por_defecto=0.7):
    """
    Solicita al usuario el umbral mínimo de correlación.
    Si no se proporciona, se utiliza un valor predeterminado.
    """
    try:
        valor = input(f"Ingrese el umbral mínimo de correlación (0-1, predeterminado {valor_por_defecto}): ")
        umbral = float(valor) if valor else valor_por_defecto
        if not 0 <= umbral <= 1:
            raise ValueError("El umbral de correlación debe estar entre 0 y 1.")
        return umbral
    except ValueError:
        print("Entrada inválida. Usando el valor predeterminado.")
        return valor_por_defecto

def solicitar_confianza_min_correlacion(valor_por_defecto=0.5):
    """
    Solicita al usuario el nivel mínimo de confianza para correlación.
    Si no se proporciona, se utiliza un valor predeterminado.
    """
    try:
        valor = input(f"Ingrese el nivel mínimo de confianza para correlación (0-1, predeterminado {valor_por_defecto}): ")
        confianza = float(valor) if valor else valor_por_defecto
        if not 0 <= confianza <= 1:
            raise ValueError("El nivel de confianza debe estar entre 0 y 1.")
        return confianza
    except ValueError:
        print("Entrada inválida. Usando el valor predeterminado.")
        return valor_por_defecto
