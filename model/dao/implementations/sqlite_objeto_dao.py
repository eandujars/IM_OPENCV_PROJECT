from typing import Any

from model.db_connection import get_db_connection
from model.dao.interfaces.objeto_dao import ObjetoDAO


class SQLiteObjetoDAO(ObjetoDAO):
    def listar(self) -> list[dict[str, Any]]:
        with get_db_connection() as connection:
            rows = connection.execute(
                """
                SELECT objeto_id, nombre, referencia, descripcion, nombre_archivo, fecha_creacion,
                       ancho_mm, alto_mm, area_mm2, perimetro_mm
                FROM objetos ORDER BY objeto_id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def buscar(
        self, consulta: str, campo: str = "todos", orden: str = "recientes"
    ) -> list[dict[str, Any]]:
        termino = f"%{consulta.strip()}%"
        columnas = {
            "todos": "(nombre LIKE ? COLLATE NOCASE OR referencia LIKE ? COLLATE NOCASE)",
            "nombre": "nombre LIKE ? COLLATE NOCASE",
            "referencia": "referencia LIKE ? COLLATE NOCASE",
        }
        ordenes = {
            "recientes": "objeto_id DESC",
            "antiguos": "objeto_id ASC",
            "nombre": "nombre COLLATE NOCASE ASC",
        }
        filtro = columnas.get(campo, columnas["todos"])
        orden_sql = ordenes.get(orden, ordenes["recientes"])
        parametros = (termino, termino) if campo == "todos" else (termino,)
        with get_db_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT objeto_id, nombre, referencia, descripcion, nombre_archivo, fecha_creacion,
                       ancho_mm, alto_mm, area_mm2, perimetro_mm
                FROM objetos
                WHERE {filtro}
                ORDER BY {orden_sql}
                """,
                parametros,
            ).fetchall()
        return [dict(row) for row in rows]

    def obtener(self, objeto_id: int) -> dict[str, Any] | None:
        with get_db_connection() as connection:
            row = connection.execute(
                "SELECT * FROM objetos WHERE objeto_id = ?", (objeto_id,)
            ).fetchone()
        return dict(row) if row else None

    def guardar(self, objeto: dict[str, Any]) -> int:
        with get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO objetos (
                    nombre, referencia, descripcion, nombre_archivo, ancho_mm, alto_mm, area_mm2,
                    perimetro_mm, ancho_px, alto_px, angulo, centro_x, centro_y,
                    imagen_original, imagen_color, imagen_bordes, imagen_a4, imagen_folio,
                    imagen_mascara, imagen_resultado
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    objeto["nombre"], objeto["referencia"], objeto["descripcion"], objeto["nombre_archivo"],
                    objeto["ancho_mm"], objeto["alto_mm"],
                    objeto["area_mm2"], objeto["perimetro_mm"],
                    objeto["ancho_px"], objeto["alto_px"], objeto["angulo"],
                    objeto["centro_x"], objeto["centro_y"],
                    objeto["imagen_original"], objeto["imagen_color"], objeto["imagen_bordes"], objeto["imagen_a4"],
                    objeto["imagen_folio"], objeto["imagen_mascara"],
                    objeto["imagen_resultado"],
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def actualizar(self, objeto_id: int, datos: dict[str, Any]) -> None:
        campos = ["nombre = ?", "referencia = ?", "descripcion = ?"]
        valores: list[Any] = [datos["nombre"], datos["referencia"], datos["descripcion"]]
        for campo in (
            "nombre_archivo", "ancho_mm", "alto_mm", "area_mm2", "perimetro_mm",
            "ancho_px", "alto_px", "angulo", "centro_x", "centro_y",
            "imagen_original", "imagen_a4", "imagen_folio", "imagen_mascara",
            "imagen_resultado", "imagen_color", "imagen_bordes",
            "anotaciones",
        ):
            if campo in datos:
                campos.append(f"{campo} = ?")
                valores.append(datos[campo])

        valores.append(objeto_id)
        with get_db_connection() as connection:
            connection.execute(
                f"UPDATE objetos SET {', '.join(campos)} WHERE objeto_id = ?",
                valores,
            )
            connection.commit()

    def eliminar(self, objeto_id: int) -> None:
        with get_db_connection() as connection:
            connection.execute("DELETE FROM objetos WHERE objeto_id = ?", (objeto_id,))
            connection.commit()