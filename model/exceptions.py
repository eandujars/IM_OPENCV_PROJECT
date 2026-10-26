class EntityNotFoundError(Exception):
    """Excepción lanzada cuando no se encuentra una entidad en la base de datos."""
    def __init__(self, message: str):
        super().__init__(message)
