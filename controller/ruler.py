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


def _interseccion_rectas(p1, d1, p2, d2):
    """Devuelve la intersección de dos rectas parametrizadas."""
    matriz = np.column_stack((d1, -d2))
    if abs(np.linalg.det(matriz)) < 1e-6:
        return None
    t = np.linalg.solve(matriz, p2 - p1)
    return p1 + t[0] * d1


# ============================================================
# DETECCIÓN DEL FOLIO A4
# ============================================================

def detectar_folio_a4(imagen: np.ndarray):
    """Detecta el A4 en la imagen y devuelve sus esquinas y contorno."""
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
# MODELADO DE FONDO Y SEGMENTACIÓN ROBUSTA
# ============================================================

def _estimar_plano_fondo(canal: np.ndarray, mascara_valida: np.ndarray) -> np.ndarray:
    """Estima un plano lineal de iluminación descartando valores atípicos."""
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
    """Segmenta objetos evitando sombras difusas y soportando transparencias."""
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

    # Cierre morfológico vertical para unir piezas de objetos alargados (como bolígrafos)
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
    """Selecciona el contorno exterior válido más grande."""
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
    """Calcula medidas métricas reales del objeto."""
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

def procesar_imagen_web(datos_imagen: bytes):
    """Ejecuta el pipeline completo y retorna resultados estructurados con todas las imágenes de depuración."""
    imagen = cv2.imdecode(np.frombuffer(datos_imagen, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError("No se pudo decodificar la imagen.")

    esquinas, contorno_a4 = detectar_folio_a4(imagen)
    if esquinas is None:
        raise ValueError("No se ha podido detectar el folio A4 de referencia.")

    img_a4 = imagen.copy()
    cv2.drawContours(img_a4, [contorno_a4.astype(np.int32)], -1, (0, 255, 0), 3)

    folio = corregir_perspectiva(imagen, esquinas)
    mascara, imagen_color_bruto, mascara_bruto = segmentar_objeto(folio)
    contorno_obj, info_contornos = detectar_objeto(mascara)

    if contorno_obj is None:
        raise ValueError("No se ha podido detectar ningún objeto sobre el folio.")

    medidas = medir_objeto(contorno_obj, info_contornos, escala=ESCALA_DEFAULT)

    resultado = folio.copy()
    cv2.drawContours(resultado, [contorno_obj], -1, (0, 255, 0), 3)
    cv2.drawContours(resultado, [medidas["caja"].reshape(-1, 1, 2)], -1, (255, 0, 0), 2)

    # Dibuja las dimensiones (ancho y alto en mm) sobre la imagen de resultado
    x_text = max(10, int(medidas["centro"][0] - 90))
    y_text = int(medidas["y"] - 15)
    if y_text < 30:
        y_text = int(medidas["y"] + medidas["alto_px"] + 30)

    texto_dimensiones = f"{medidas['ancho_mm']:.1f} x {medidas['alto_mm']:.1f} mm"
    # Texto con borde blanco y relleno rojo para máxima visibilidad
    cv2.putText(resultado, texto_dimensiones, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(resultado, texto_dimensiones, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    gris = cv2.cvtColor(folio, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)

    return {
        "medidas": medidas,
        "imagenes": {
            "original": datos_imagen,
            "a4": cv2.imencode('.jpg', img_a4)[1].tobytes(),
            "folio": cv2.imencode('.jpg', folio)[1].tobytes(),
            "color": cv2.imencode('.png', imagen_color_bruto)[1].tobytes(),
            "bordes": cv2.imencode('.png', bordes)[1].tobytes(),
            "mascara": cv2.imencode('.png', mascara)[1].tobytes(),
            "resultado": cv2.imencode('.jpg', resultado)[1].tobytes(),
        }
    }


__all__ = [
    "A4_ANCHO_MM",
    "A4_ALTO_MM",
    "ESCALA_DEFAULT",
    "A4_ANCHO_PX",
    "A4_ALTO_PX",
    "detectar_folio_a4",
    "corregir_perspectiva",
    "segmentar_objeto",
    "detectar_objeto",
    "medir_objeto",
    "procesar_imagen_web",
]