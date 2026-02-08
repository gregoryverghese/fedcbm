"""Base I/O interface for MINOTAUR data backends."""
from abc import ABC, abstractmethod
from typing import List, Any, Optional


class IOBackend(ABC):
    """Abstract base class for I/O backends."""
    
    @abstractmethod
    def get_keys(self) -> List[bytes]:
        """Get all keys in the database."""
        pass
    
    @abstractmethod
    def read_ndarray(self, key: bytes) -> Any:
        """Read an array from the database."""
        pass
    
    @property
    @abstractmethod
    def num_keys(self) -> int:
        """Get the number of keys in the database."""
        pass


