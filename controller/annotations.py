import cv2
import numpy as np


ANCHO_PIZARRA = 900
ALTO_PIZARRA = 500


def crear_pizarra() -> bytes:
    """Crea una pizarra blanca PNG usando OpenCV."""
    pizarra = np.full(
        (ALTO_PIZARRA, ANCHO_PIZARRA, 3),
        255,
        dtype=np.uint8,
    )
    ok, buffer = cv2.imencode(".png", pizarra)
    if not ok:
        raise ValueError("No se pudo crear la pizarra.")
    return buffer.tobytes()


def normalizar_anotacion(datos: bytes) -> bytes:
    """Valida y normaliza una anotación dibujada antes de guardarla."""
    imagen = cv2.imdecode(np.frombuffer(datos, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise ValueError("La anotación no contiene una imagen válida.")

    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    elif imagen.shape[2] == 4:
        fondo = np.full(imagen.shape[:2] + (3,), 255, dtype=np.uint8)
        alpha = imagen[:, :, 3:4].astype(np.float32) / 255
        color = imagen[:, :, :3].astype(np.float32)
        imagen = (color * alpha + fondo * (1 - alpha)).astype(np.uint8)

    imagen = cv2.resize(
        imagen,
        (ANCHO_PIZARRA, ALTO_PIZARRA),
        interpolation=cv2.INTER_AREA,
    )
    ok, buffer = cv2.imencode(".png", imagen)
    if not ok:
        raise ValueError("No se pudo codificar la anotación.")
    return buffer.tobytes()
