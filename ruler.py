import os
import sys
import glob
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
# CLASE VISOR INTERACTIVO (ZOOM, SCROLL Y AJUSTAR)
# ============================================================

class VisorVentana:
    """Maneja el zoom, scroll y ajuste automático para una ventana"""
    def __init__(self, nombre_ventana, imagen, win_w=900, win_h=850):
        self.nombre_ventana = nombre_ventana
        
        # Convertir a 3 canales BGR si la imagen es en escala de grises / máscara
        if len(imagen.shape) == 2:
            self.img_orig = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            self.img_orig = imagen.copy()
            
        self.h, self.w = self.img_orig.shape[:2]

        # Ventanas de tamaño fijo y apaisado para evitar ventanas muy altas.
        # La imagen se adapta dentro de la ventana conservando su proporción.
        # Esto solo afecta a la visualización, no a la imagen usada para medir.
        self.win_w = 900
        self.win_h = 700
        
        # Estado del arrastre (Scroll)
        self.dragging = False
        self.drag_start = (0, 0)
        self.roi_start = (0.0, 0.0, 0.0, 0.0)
        
        # Ajustar vista inicial
        self.ajustar()

    def ajustar(self):
        """Restablece la vista para ajustar la imagen completa a la ventana."""
        self.x1 = 0.0
        self.y1 = 0.0
        self.x2 = float(self.w)
        self.y2 = float(self.h)

    def zoom_at(self, wx, wy, factor):
        """Aplica zoom in/out centrado en la posición del ratón."""
        ix = self.x1 + (wx / float(self.win_w)) * (self.x2 - self.x1)
        iy = self.y1 + (wy / float(self.win_h)) * (self.y2 - self.y1)
        
        curr_w = self.x2 - self.x1
        curr_h = self.y2 - self.y1
        
        new_w = curr_w / factor
        new_h = curr_h / factor
        
        # Si el zoom out supera el tamaño original, ajustar
        if new_w >= self.w or new_h >= self.h:
            self.ajustar()
            return
            
        if new_w < 10 or new_h < 10:
            return
            
        rx = wx / float(self.win_w)
        ry = wy / float(self.win_h)
        
        nx1 = ix - rx * new_w
        nx2 = nx1 + new_w
        ny1 = iy - ry * new_h
        ny2 = ny1 + new_h
        
        # Limitar dentro de los bordes de la imagen
        if nx1 < 0:
            nx2 -= nx1
            nx1 = 0.0
        if nx2 > self.w:
            nx1 -= (nx2 - self.w)
            nx2 = float(self.w)
        if ny1 < 0:
            ny2 -= ny1
            ny1 = 0.0
        if ny2 > self.h:
            ny1 -= (ny2 - self.h)
            ny2 = float(self.h)
            
        self.x1, self.x2 = max(0.0, nx1), min(float(self.w), nx2)
        self.y1, self.y2 = max(0.0, ny1), min(float(self.h), ny2)

    def mover(self, dwx, dwy):
        """Desplaza la imagen (Scroll) en función del arrastre del ratón."""
        scale_x = (self.roi_start[2] - self.roi_start[0]) / float(self.win_w)
        scale_y = (self.roi_start[3] - self.roi_start[1]) / float(self.win_h)
        
        d_ix = dwx * scale_x
        d_iy = dwy * scale_y
        
        nx1 = self.roi_start[0] - d_ix
        nx2 = self.roi_start[2] - d_ix
        ny1 = self.roi_start[1] - d_iy
        ny2 = self.roi_start[3] - d_iy
        
        if nx1 < 0:
            nx2 += (0 - nx1)
            nx1 = 0.0
        if nx2 > self.w:
            nx1 -= (nx2 - self.w)
            nx2 = float(self.w)
        if ny1 < 0:
            ny2 += (0 - ny1)
            ny1 = 0.0
        if ny2 > self.h:
            ny1 -= (ny2 - self.h)
            ny2 = float(self.h)
            
        self.x1, self.x2 = max(0.0, nx1), min(float(self.w), nx2)
        self.y1, self.y2 = max(0.0, ny1), min(float(self.h), ny2)

    def obtener_vista(self):
        """Devuelve el recorte de la imagen escalado a la ventana con la barra de ayuda."""
        ix1, iy1 = int(round(self.x1)), int(round(self.y1))
        ix2, iy2 = int(round(self.x2)), int(round(self.y2))
        
        ix1 = max(0, min(self.w - 1, ix1))
        iy1 = max(0, min(self.h - 1, iy1))
        ix2 = max(ix1 + 1, min(self.w, ix2))
        iy2 = max(iy1 + 1, min(self.h, iy2))
        
        recorte = self.img_orig[iy1:iy2, ix1:ix2]
        # Reservar la franja inferior para la ayuda y conservar la proporción.
        area_h = self.win_h - 22
        escala = min(self.win_w / recorte.shape[1], area_h / recorte.shape[0])
        nuevo_w = max(1, int(round(recorte.shape[1] * escala)))
        nuevo_h = max(1, int(round(recorte.shape[0] * escala)))
        imagen_redimensionada = cv2.resize(
            recorte, (nuevo_w, nuevo_h), interpolation=cv2.INTER_LINEAR
        )
        vista = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        x_offset = (self.win_w - nuevo_w) // 2
        vista[0:nuevo_h, x_offset:x_offset + nuevo_w] = imagen_redimensionada
        
        # Barra de ayuda rápida inferior
        cv2.rectangle(vista, (0, self.win_h - 22), (self.win_w, self.win_h), (30, 30, 30), -1)
        cv2.putText(vista, "Rueda/+-: Zoom | Arrastrar: Scroll | Doble clic/'A': Ajustar", 
                    (8, self.win_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        return vista


def mouse_callback(event, x, y, flags, param):
    """Callback de ratón"""
    visor = param
    if event == cv2.EVENT_LBUTTONDOWN:
        visor.dragging = True
        visor.drag_start = (x, y)
        visor.roi_start = (visor.x1, visor.y1, visor.x2, visor.y2)
    elif event == cv2.EVENT_MOUSEMOVE and visor.dragging:
        dwx = x - visor.drag_start[0]
        dwy = y - visor.drag_start[1]
        visor.mover(dwx, dwy)
    elif event == cv2.EVENT_LBUTTONUP:
        visor.dragging = False
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        visor.ajustar()
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            visor.zoom_at(x, y, 1.25)
        else:
            visor.zoom_at(x, y, 0.8)


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
# MODELADO DE FONDO Y SEGMENTACIÓN
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
# VISUALIZACIÓN INTERACTIVA DE IMÁGENES Y MÁSCARAS
# ============================================================

def procesar_imagen_especifica(ruta_imagen: str):
    """Carga, procesa y abre las ventanas con zoom, scroll y fit interactivo."""
    nombre_archivo = os.path.basename(ruta_imagen)
    imagen = cv2.imread(ruta_imagen)
    
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen '{ruta_imagen}'.")
        return

    print(f"\n--- Procesando: {nombre_archivo} ---")
    esquinas, contorno_a4 = detectar_folio_a4(imagen)
    if esquinas is None:
        print("Error: No se ha podido detectar el folio A4 en la imagen.")
        cv2.imshow("Imagen Original (A4 no detectado)", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    img_a4 = imagen.copy()
    cv2.drawContours(img_a4, [contorno_a4.astype(np.int32)], -1, (0, 255, 0), 3)

    folio = corregir_perspectiva(imagen, esquinas)
    mascara_relleno, m_color, mascara_bruta = segmentar_objeto(folio)
    contorno_obj, info_contornos = detectar_objeto(mascara_relleno)

    gris = cv2.cvtColor(folio, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)

    resultado = folio.copy()

    if contorno_obj is not None:
        medidas = medir_objeto(contorno_obj, info_contornos, escala=ESCALA_DEFAULT)
        cv2.drawContours(resultado, [contorno_obj], -1, (0, 255, 0), 3)
        cv2.drawContours(resultado, [medidas["caja"].reshape(-1, 1, 2)], -1, (255, 0, 0), 2)

        x_text = max(10, int(medidas["centro"][0] - 90))
        y_text = int(medidas["y"] - 15)
        if y_text < 30:
            y_text = int(medidas["y"] + medidas["alto_px"] + 30)

        texto = f"{medidas['ancho_mm']:.1f} x {medidas['alto_mm']:.1f} mm"
        cv2.putText(resultado, texto, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(resultado, texto, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        print(f"-> Dimensiones: {medidas['ancho_mm']:.1f} x {medidas['alto_mm']:.1f} mm")
        print(f"-> Área: {medidas['area_mm2']:.1f} mm² | Perímetro: {medidas['perimetro_mm']:.1f} mm")
    else:
        print("-> Folio detectado, pero no se encontró un objeto válido.")

    # Crear visores interactivos para cada una de las máscaras e imágenes intermedias
    imagenes_dict = {
        f"1. A4 Detectado ({nombre_archivo})": img_a4,
        "2. Folio Corregido": folio,
        "3. Mascara Color": m_color,
        "4. Bordes Canny": bordes,
        "5. Mascara Objeto Final": mascara_relleno,
        "6. Resultado Medicion": resultado,
    }

    visores = []
    for nombre_win, img in imagenes_dict.items():
        v = VisorVentana(nombre_win, img)
        cv2.namedWindow(nombre_win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(nombre_win, mouse_callback, v)
        visores.append(v)

    print("\n[Controles Interactivos]")
    print(" - Rueda del ratón / Teclas '+' y '-': Zoom In / Zoom Out")
    print(" - Clic izquierdo y arrastrar: Scroll / Desplazar imagen")
    print(" - Doble Clic / Teclas 'A', 'R' o Espacio: Ajustar vista (Reset Zoom)")
    print(" - Tecla 'Q' o ESC: Cerrar todas las ventanas")

    # Bucle de renderizado continuo para responder a ratón y teclado
    while True:
        for v in visores:
            cv2.imshow(v.nombre_ventana, v.obtener_vista())

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('+'), ord('=')):
            for v in visores:
                v.zoom_at(v.win_w // 2, v.win_h // 2, 1.25)
        elif key in (ord('-'), ord('_')):
            for v in visores:
                v.zoom_at(v.win_w // 2, v.win_h // 2, 0.8)
        elif key in (ord('a'), ord('A'), ord('r'), ord('R'), 32):  # 32 es Espacio
            for v in visores:
                v.ajustar()

    cv2.destroyAllWindows()


def menu_seleccion(directorio_imagenes="imagenes"):
    """Selecciona la imagen mediante argumento de consola o menú interactivo."""
    if len(sys.argv) > 1:
        param = sys.argv[1]
        ruta = param if os.path.exists(param) else os.path.join(directorio_imagenes, param)
        if os.path.exists(ruta):
            procesar_imagen_especifica(ruta)
        else:
            print(f"Error: No existe el archivo '{ruta}'.")
        return

    extensiones = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    archivos = []
    for ext in extensiones:
        archivos.extend(glob.glob(os.path.join(directorio_imagenes, ext)))
        archivos.extend(glob.glob(os.path.join(directorio_imagenes, ext.upper())))

    archivos = sorted(list(set(archivos)))

    if not archivos:
        print(f"No se encontraron imágenes dentro de la carpeta '{directorio_imagenes}'.")
        return

    print("\n" + "=" * 45)
    print("      SELECCIONA LA IMAGEN A PROCESAR")
    print("=" * 45)
    for idx, ruta in enumerate(archivos, 1):
        print(f"  [{idx}] {os.path.basename(ruta)}")
    print("=" * 45)

    try:
        entrada = input(f"Introduce un número (1-{len(archivos)}) o el nombre del archivo: ").strip()
        
        if entrada.isdigit():
            num = int(entrada) - 1
            if 0 <= num < len(archivos):
                procesar_imagen_especifica(archivos[num])
            else:
                print("Número fuera de rango.")
        else:
            ruta_directa = os.path.join(directorio_imagenes, entrada)
            if os.path.exists(ruta_directa):
                procesar_imagen_especifica(ruta_directa)
            else:
                print(f"El archivo '{entrada}' no existe.")
    except (KeyboardInterrupt, EOFError):
        print("\nCancelado.")


if __name__ == "__main__":
    menu_seleccion("imagenes")