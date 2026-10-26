from pathlib import Path
from typing import Any

from controller.ruler import procesar_imagen_web
from model.dao.implementations.sqlite_objeto_dao import SQLiteObjetoDAO


class ObjetoModel:
	def __init__(self, dao: SQLiteObjetoDAO | None = None):
		self.dao = dao or SQLiteObjetoDAO()

	def listar(self) -> list[dict[str, Any]]:
		return self.dao.listar()

	def obtener(self, objeto_id: int) -> dict[str, Any] | None:
		return self.dao.obtener(objeto_id)

	def buscar(
		self, consulta: str, campo: str = "todos", orden: str = "recientes"
	) -> list[dict[str, Any]]:
		return self.dao.buscar(consulta, campo, orden)

	def procesar_y_guardar(
		self, nombre: str, referencia: str, descripcion: str, nombre_archivo: str, datos: bytes
	) -> int:
		resultado = procesar_imagen_web(datos)
		medidas = resultado["medidas"]
		nombre = nombre.strip() or Path(nombre_archivo).stem.replace("_", " ").strip() or "Objeto sin nombre"
		imagenes = resultado["imagenes"]
		return self.dao.guardar({
			"nombre": nombre,
			"referencia": referencia.strip(),
			"descripcion": descripcion.strip(),
			"nombre_archivo": nombre_archivo,
			"ancho_mm": float(medidas["ancho_mm"]),
			"alto_mm": float(medidas["alto_mm"]),
			"area_mm2": float(medidas["area_mm2"]),
			"perimetro_mm": float(medidas["perimetro_mm"]),
			"ancho_px": float(medidas["ancho_px"]),
			"alto_px": float(medidas["alto_px"]),
			"angulo": float(medidas["angulo"]),
			"centro_x": float(medidas["centro"][0]),
			"centro_y": float(medidas["centro"][1]),
			"imagen_original": imagenes["original"],
			"imagen_color": imagenes["color"],
			"imagen_bordes": imagenes["bordes"],
			"imagen_a4": imagenes["a4"],
			"imagen_folio": imagenes["folio"],
			"imagen_mascara": imagenes["mascara"],
			"imagen_resultado": imagenes["resultado"],
		})

	def actualizar(
		self, objeto_id: int, nombre: str, referencia: str, descripcion: str,
		nombre_archivo: str | None = None, datos: bytes | None = None,
	) -> None:
		objeto = self.obtener(objeto_id)
		if objeto is None:
			raise ValueError("El objeto no existe.")

		cambios: dict[str, Any] = {
			"nombre": nombre.strip(),
			"referencia": referencia.strip(),
			"descripcion": descripcion.strip(),
		}
		if datos:
			resultado = procesar_imagen_web(datos)
			medidas = resultado["medidas"]
			imagenes = resultado["imagenes"]
			cambios.update({
				"nombre_archivo": nombre_archivo or objeto["nombre_archivo"],
				"ancho_mm": float(medidas["ancho_mm"]),
				"alto_mm": float(medidas["alto_mm"]),
				"area_mm2": float(medidas["area_mm2"]),
				"perimetro_mm": float(medidas["perimetro_mm"]),
				"ancho_px": float(medidas["ancho_px"]),
				"alto_px": float(medidas["alto_px"]),
				"angulo": float(medidas["angulo"]),
				"centro_x": float(medidas["centro"][0]),
				"centro_y": float(medidas["centro"][1]),
				"imagen_original": imagenes["original"],
				"imagen_color": imagenes["color"],
				"imagen_bordes": imagenes["bordes"],
				"imagen_a4": imagenes["a4"],
				"imagen_folio": imagenes["folio"],
				"imagen_mascara": imagenes["mascara"],
				"imagen_resultado": imagenes["resultado"],
			})
		self.dao.actualizar(objeto_id, cambios)

	def eliminar(self, objeto_id: int) -> None:
		self.dao.eliminar(objeto_id)

	def guardar_anotaciones(self, objeto_id: int, imagen: bytes) -> None:
		objeto = self.obtener(objeto_id)
		if objeto is None:
			raise ValueError("El objeto no existe.")
		self.dao.actualizar(
			objeto_id,
			{
				"nombre": objeto["nombre"],
				"referencia": objeto["referencia"],
				"descripcion": objeto["descripcion"],
				"anotaciones": imagen,
			},
		)

	def obtener_color(self, objeto_id: int) -> bytes | None:
		objeto = self.obtener(objeto_id)
		if objeto is None:
			return None
		if objeto.get("imagen_color"):
			return objeto["imagen_color"]

		resultado = procesar_imagen_web(objeto["imagen_original"])
		color = resultado["imagenes"]["color"]
		self.dao.actualizar(
			objeto_id,
			{
				"nombre": objeto["nombre"],
				"referencia": objeto["referencia"],
				"descripcion": objeto["descripcion"],
				"imagen_color": color,
			},
		)
		return color

	def completar_medidas(self, objeto_id: int) -> None:
		objeto = self.obtener(objeto_id)
		if objeto is None or (
			objeto.get("ancho_px") is not None and objeto.get("imagen_bordes")
		):
			return

		resultado = procesar_imagen_web(objeto["imagen_original"])
		medidas = resultado["medidas"]
		self.dao.actualizar(
			objeto_id,
			{
				"nombre": objeto["nombre"],
				"referencia": objeto["referencia"],
				"descripcion": objeto["descripcion"],
				"ancho_mm": float(medidas["ancho_mm"]),
				"alto_mm": float(medidas["alto_mm"]),
				"area_mm2": float(medidas["area_mm2"]),
				"perimetro_mm": float(medidas["perimetro_mm"]),
				"ancho_px": float(medidas["ancho_px"]),
				"alto_px": float(medidas["alto_px"]),
				"angulo": float(medidas["angulo"]),
				"centro_x": float(medidas["centro"][0]),
				"centro_y": float(medidas["centro"][1]),
				"imagen_color": resultado["imagenes"]["color"],
				"imagen_bordes": resultado["imagenes"]["bordes"],
			},
		)
