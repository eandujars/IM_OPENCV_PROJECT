from abc import ABC, abstractmethod
from typing import Any


class ObjetoDAO(ABC):
    @abstractmethod
    def listar(self) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def obtener(self, objeto_id: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def buscar(self, consulta: str) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def guardar(self, objeto: dict[str, Any]) -> int:
        raise NotImplementedError

    @abstractmethod
    def actualizar(self, objeto_id: int, datos: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def eliminar(self, objeto_id: int) -> None:
        raise NotImplementedError