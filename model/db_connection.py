import sqlite3
from pathlib import Path
from sqlite3 import Connection

DATABASE_PATH = str(Path(__file__).with_name("midb.sqlite3"))

def get_db_connection() -> Connection:
    """Crea y devuelve una conexión a la base de datos SQLite."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Permite acceder a las columnas por nombre
    return conn