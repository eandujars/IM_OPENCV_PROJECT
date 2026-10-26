import os
import json
import sqlite3  
from model.db_connection import DATABASE_PATH, get_db_connection

def initialize_database():
    """Inicializa la base de datos con el esquema definido y carga datos iniciales desde un JSON
    solo si la base de datos aún no existe."""

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    try:
        with get_db_connection() as conn:
            with open(schema_path, "r", encoding="utf-8") as schema_file:
                conn.executescript(schema_file.read())
            columnas = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(objetos)").fetchall()
            }
            if "referencia" not in columnas:
                conn.execute(
                    "ALTER TABLE objetos ADD COLUMN referencia TEXT NOT NULL DEFAULT ''"
                )
                conn.commit()
            if "descripcion" not in columnas:
                conn.execute(
                    "ALTER TABLE objetos ADD COLUMN descripcion TEXT NOT NULL DEFAULT ''"
                )
                conn.commit()
            if "anotaciones" not in columnas:
                conn.execute("ALTER TABLE objetos ADD COLUMN anotaciones BLOB")
                conn.commit()
            if "imagen_color" not in columnas:
                conn.execute("ALTER TABLE objetos ADD COLUMN imagen_color BLOB")
                conn.commit()
            if "imagen_bordes" not in columnas:
                conn.execute("ALTER TABLE objetos ADD COLUMN imagen_bordes BLOB")
                conn.commit()
            for columna in ("ancho_px", "alto_px", "angulo", "centro_x", "centro_y"):
                if columna not in columnas:
                    conn.execute(f"ALTER TABLE objetos ADD COLUMN {columna} REAL")
                    conn.commit()
    except (OSError, sqlite3.Error) as error:
        raise RuntimeError(f"No se pudo inicializar la base de datos: {error}") from error