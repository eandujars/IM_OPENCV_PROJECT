import cv2
import numpy as np

# ============================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================

A4_ANCHO_MM = 210.0
A4_ALTO_MM = 297.0

# Normalización de perspectiva a 5 px/mm (1050 x 1485 px)
ESCALA_DEFAULT = 5.0
A4_ANCHO_PX = int(A4_ANCHO_MM * ESCALA_DEFAULT)
A4_ALTO_PX = int(A4_ALTO_MM * ESCALA_DEFAULT)

# Umbrales por defecto ajustables
UMBRAL_COLOR_DEFAULT = 12.0
UMBRAL_OSCURO_DEFAULT = 100.0
UMBRAL_CLARO_DEFAULT = 10.0


# ============================================================
# UTILIDADES GEOMÉTRICAS
# ============================================================

def ordenar_puntos(puntos: np.ndarray) -> np.ndarray:
    """Ordena cuatro puntos como: TL, TR, BR, BL."""
    puntos = np.asarray(puntos, dtype=np.float32)
    if puntos.shape != (4, 2):
        raise ValueError("Se esperaban exactamente cuatro puntos.")

    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1).ravel()

    return np.array([
        puntos[np.argmin(suma)],       # arriba-izquierda
        puntos[np.argmin(diferencia)], # arriba-derecha
        puntos[np.argmax(suma)],       # abajo-derecha
        puntos[np.argmax(diferencia)], # abajo-izquierda
    ], dtype=np.float32)


# ============================================================
# DETECCIÓN DEL FOLIO A4
# ============================================================

def detectar_folio_a4(imagen: np.ndarray):
    """Detecta el A4 en la imagen y devuelve sus esquinas y contorno"""
    alto, ancho = imagen.shape[:2]
    suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)
    lab = cv2.cvtColor(suavizada, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    croma = np.hypot(A - 128.0, B - 128.0)
    croma8 = np.clip(croma * 4.0, 0, 255).astype(np.uint8)
    t_croma, _ = cv2.threshold(croma8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    umbral_croma = float(np.clip(t_croma / 4.0, 12.0, 25.0))
    sin_color = croma < umbral_croma

    mascara = (sin_color & (L > 20.0)).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None

    contorno_principal = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno_principal) < 0.10 * alto * ancho:
        return None, None

    casco = cv2.convexHull(contorno_principal)
    perimetro = cv2.arcLength(casco, True)
    puntos = None

    for factor in (0.02, 0.03, 0.04, 0.05, 0.08):
        aprox = cv2.approxPolyDP(casco, factor * perimetro, True)
        if len(aprox) == 4:
            puntos = aprox.reshape(4, 2)
            break

    if puntos is None:
        return None, None

    puntos = ordenar_puntos(puntos)
    
    ancho_a4 = np.linalg.norm(puntos[1] - puntos[0])
    alto_a4 = np.linalg.norm(puntos[3] - puntos[0])
    if ancho_a4 > alto_a4:
        puntos = np.roll(puntos, -1, axis=0)

    return puntos, contorno_principal


def detectar_folio_centrado(datos_imagen: bytes) -> bool:
    """Comprueba si un frame de cámara contiene el A4 dentro de la guía."""
    try:
        imagen = cv2.imdecode(
            np.frombuffer(datos_imagen, dtype=np.uint8), cv2.IMREAD_COLOR
        )
    except cv2.error:
        return False

    if imagen is None:
        return False

    try:
        esquinas, _ = detectar_folio_a4(imagen)
    except cv2.error:
        return False
    if esquinas is None:
        return False

    alto, ancho = imagen.shape[:2]
    escala_x = ancho / 1280.0
    escala_y = alto / 720.0
    guia = np.array([
        [(1280 - 420) / 2, (720 - 594) / 2],
        [(1280 + 420) / 2, (720 - 594) / 2],
        [(1280 + 420) / 2, (720 + 594) / 2],
        [(1280 - 420) / 2, (720 + 594) / 2],
    ], dtype=np.float32)
    guia[:, 0] *= escala_x
    guia[:, 1] *= escala_y
    distancias = np.linalg.norm(
        ordenar_puntos(esquinas) - guia, axis=1
    )
    return bool(np.max(distancias) <= 35 * max(escala_x, escala_y))


def corregir_perspectiva(imagen: np.ndarray, esquinas: np.ndarray) -> np.ndarray:
    """Transforma el A4 detectado a un lienzo normalizado."""
    destino = np.array([
        [0, 0],
        [A4_ANCHO_PX - 1, 0],
        [A4_ANCHO_PX - 1, A4_ALTO_PX - 1],
        [0, A4_ALTO_PX - 1],
    ], dtype=np.float32)

    matriz = cv2.getPerspectiveTransform(ordenar_puntos(esquinas), destino)
    return cv2.warpPerspective(imagen, matriz, (A4_ANCHO_PX, A4_ALTO_PX))


# ============================================================
# MODELADO DE FONDO Y SEGMENTACIÓN
# ============================================================

def _estimar_plano_fondo(canal: np.ndarray, mascara_valida: np.ndarray) -> np.ndarray:
    """Estima un plano lineal de iluminación descartando valores atípicos"""
    ys, xs = np.nonzero(mascara_valida)
    if len(xs) < 50:
        return np.full_like(canal, np.median(canal), dtype=np.float32)

    ys, xs = ys[::5], xs[::5]
    valores = canal[ys, xs].astype(np.float64)
    matriz = np.column_stack((xs, ys, np.ones_like(xs))).astype(np.float64)

    coef = np.zeros(3, dtype=np.float64)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(matriz, valores, rcond=None)
        residuo = valores - matriz @ coef
        mad = np.median(np.abs(residuo - np.median(residuo))) + 1e-6
        validos = np.abs(residuo) < 3.5 * 1.4826 * mad
        if not np.any(validos):
            break
        matriz = matriz[validos]
        valores = valores[validos]

    X, Y = np.meshgrid(
        np.arange(canal.shape[1], dtype=np.float32),
        np.arange(canal.shape[0], dtype=np.float32),
    )
    return (coef[0] * X + coef[1] * Y + coef[2]).astype(np.float32)


def segmentar_objeto(imagen: np.ndarray):
    """Segmenta objetos evitando sombras difusas y soportando transparencias"""
    alto, ancho = imagen.shape[:2]
    
    suavizada = cv2.GaussianBlur(imagen, (7, 7), 0)
    lab = cv2.cvtColor(suavizada, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    margen = int(0.02 * min(alto, ancho))
    mascara_fondo = np.zeros((alto, ancho), dtype=np.uint8)
    mascara_fondo[margen:-margen, margen:-margen] = 255

    fondo_L = _estimar_plano_fondo(L, mascara_fondo)
    fondo_A = _estimar_plano_fondo(A, mascara_fondo)
    fondo_B = _estimar_plano_fondo(B, mascara_fondo)

    dL = fondo_L - L
    dC = np.hypot(A - fondo_A, B - fondo_B)

    ruido_L = 1.4826 * np.median(np.abs(dL[mascara_fondo > 0] - np.median(dL[mascara_fondo > 0])))
    ruido_C = 1.4826 * np.median(np.abs(dC[mascara_fondo > 0] - np.median(dC[mascara_fondo > 0])))

    umbral_color = max(UMBRAL_COLOR_DEFAULT, float(ruido_C * 4.0))
    umbral_claro = max(UMBRAL_CLARO_DEFAULT, float(ruido_L * 5.0))

    m_color = (dC > umbral_color).astype(np.uint8) * 255
    m_oscuro = (dL > UMBRAL_OSCURO_DEFAULT).astype(np.uint8) * 255
    m_claro = (dL < -umbral_claro).astype(np.uint8) * 255

    mascara_base = cv2.bitwise_or(m_color, m_oscuro)
    mascara_base = cv2.bitwise_or(mascara_base, m_claro)

    # Cierre morfológico vertical para unir piezas de objetos alargados
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 41))
    mascara = cv2.morphologyEx(mascara_base, cv2.MORPH_CLOSE, kernel_vertical)

    mascara[:margen, :] = 0
    mascara[-margen:, :] = 0
    mascara[:, :margen] = 0
    mascara[:, -margen:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    relleno = np.zeros((alto, ancho), np.uint8)

    if jerarquia is not None and len(contornos) > 0:
        for i, contorno in enumerate(contornos):
            if cv2.contourArea(contorno) > 300:
                cv2.drawContours(relleno, [contorno], -1, 255, cv2.FILLED)

    return relleno, m_color, mascara


# ============================================================
# DETECCIÓN Y MEDICIÓN DEL OBJETO
# ============================================================

def detectar_objeto(mascara: np.ndarray):
    """Selecciona el contorno exterior válido más grande"""
    contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None

    alto, ancho = mascara.shape
    jerarquia = jerarquia[0] if jerarquia is not None else None

    mejor_indice = None
    mayor_area = 0.0

    for i, contorno in enumerate(contornos):
        if jerarquia is not None and jerarquia[i][3] != -1:
            continue

        area = cv2.contourArea(contorno)
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(contorno)
        if x <= 2 or y <= 2 or x + w >= ancho - 2 or y + h >= alto - 2:
            continue

        if area > mayor_area:
            mayor_area = area
            mejor_indice = i

    if mejor_indice is None:
        return None, None

    return contornos[mejor_indice], {
        "contornos": contornos,
        "jerarquia": jerarquia,
        "indice_externo": mejor_indice,
    }


def medir_objeto(contorno, info_contornos=None, escala=ESCALA_DEFAULT):
    """Calcula medidas métricas reales del objeto"""
    if contorno is None:
        raise ValueError("Contorno inválido para medir.")

    if info_contornos is None:
        area_px2 = cv2.contourArea(contorno)
    else:
        contornos = info_contornos["contornos"]
        jerarquia = info_contornos["jerarquia"]
        indice = info_contornos["indice_externo"]
        area_px2 = cv2.contourArea(contornos[indice])
        if jerarquia is not None:
            hijo = jerarquia[indice][2]
            while hijo != -1:
                area_px2 -= cv2.contourArea(contornos[hijo])
                hijo = jerarquia[hijo][0]
        area_px2 = max(area_px2, 0.0)

    (cx, cy), (l1, l2), angulo = cv2.minAreaRect(contorno)
    ancho_px, alto_px = min(l1, l2), max(l1, l2)
    perimetro_px = cv2.arcLength(contorno, True)
    x, y, _, _ = cv2.boundingRect(contorno)
    caja = cv2.boxPoints(((cx, cy), (l1, l2), angulo)).astype(np.int32)

    return {
        "x": int(x),
        "y": int(y),
        "ancho_px": float(ancho_px),
        "alto_px": float(alto_px),
        "ancho_mm": float(ancho_px / escala),
        "alto_mm": float(alto_px / escala),
        "area_mm2": float(area_px2 / (escala ** 2)),
        "perimetro_mm": float(perimetro_px / escala),
        "centro": (float(cx), float(cy)),
        "angulo": float(angulo),
        "caja": caja,
    }


# ============================================================
# PIPELINE PRINCIPAL WEB
# ============================================================

def _codificar_imagen(imagen: np.ndarray, extension: str) -> bytes:
    """Codifica una imagen OpenCV y convierte los fallos en errores del pipeline"""
    ok, buffer = cv2.imencode(extension, imagen)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como {extension}.")
    return buffer.tobytes()


def procesar_imagen_web(datos_imagen: bytes):
    """Procesa una imagen subida y devuelve medidas e imágenes para la interfaz web.

    La versión interactiva de ``ruler.py`` utiliza ventanas de OpenCV. En la web
    esas ventanas no son válidas: el resultado se devuelve como bytes para que
    FastAPI y las plantillas puedan mostrar cada vista en el navegador
    """
    imagen = cv2.imdecode(np.frombuffer(datos_imagen, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError("No se pudo decodificar la imagen.")

    esquinas, contorno_a4 = detectar_folio_a4(imagen)
    if esquinas is None:
        raise ValueError("No se ha podido detectar el folio A4 de referencia.")

    img_a4 = imagen.copy()
    cv2.drawContours(img_a4, [contorno_a4.astype(np.int32)], -1, (0, 255, 0), 3)

    folio = corregir_perspectiva(imagen, esquinas)
    mascara, imagen_color_bruto, _ = segmentar_objeto(folio)
    contorno_obj, info_contornos = detectar_objeto(mascara)

    if contorno_obj is None:
        raise ValueError("No se ha podido detectar ningún objeto sobre el folio.")

    medidas = medir_objeto(contorno_obj, info_contornos, escala=ESCALA_DEFAULT)

    resultado = folio.copy()
    cv2.drawContours(resultado, [contorno_obj], -1, (0, 255, 0), 3)
    cv2.drawContours(resultado, [medidas["caja"].reshape(-1, 1, 2)], -1, (255, 0, 0), 2)

    # Mantiene la etiqueta legible aunque el objeto ocupe toda la fotografía.
    texto = f"{medidas['ancho_mm']:.1f} x {medidas['alto_mm']:.1f} mm"
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    grosor = 2
    margen = 18
    (texto_ancho, texto_alto), linea_base = cv2.getTextSize(
        texto, fuente, 1.0, grosor
    )
    x1, y1 = 12, 12
    x2 = x1 + texto_ancho + margen * 2
    y2 = y1 + texto_alto + linea_base + margen
    cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        resultado,
        texto,
        (x1 + margen, y1 + margen + texto_alto),
        fuente,
        1.0,
        (255, 255, 255),
        grosor,
        cv2.LINE_AA,
    )

    gris = cv2.cvtColor(folio, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)

    return {
        "medidas": medidas,
        "imagenes": {
            "original": datos_imagen,
            "a4": _codificar_imagen(img_a4, ".jpg"),
            "folio": _codificar_imagen(folio, ".jpg"),
            "color": _codificar_imagen(imagen_color_bruto, ".png"),
            "bordes": _codificar_imagen(bordes, ".png"),
            "mascara": _codificar_imagen(mascara, ".png"),
            "resultado": _codificar_imagen(resultado, ".jpg"),
        }
    }


__all__ = [
    "A4_ANCHO_MM",
    "A4_ALTO_MM",
    "ESCALA_DEFAULT",
    "A4_ANCHO_PX",
    "A4_ALTO_PX",
    "detectar_folio_a4",
    "detectar_folio_centrado",
    "corregir_perspectiva",
    "segmentar_objeto",
    "detectar_objeto",
    "medir_objeto",
    "procesar_imagen_web",
]