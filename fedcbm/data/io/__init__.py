"""I/O backends for data loading."""
from .base import IOBackend
from .lmdb import LMDBRead, LMDBWrite, NpyObject, Tile
from .disk import DiskRead, DiskWrite
from .rocksdb import RocksDBRead, RocksDBWrite

__all__ = [
    "IOBackend",
    "LMDBRead",
    "LMDBWrite",
    "DiskRead",
    "DiskWrite",
    "RocksDBRead",
    "RocksDBWrite",
    "NpyObject",
    "Tile",
]


def get_io_backend(backend_type: str, db_path: str, mode: str = "read", **kwargs):
    """
    Factory function to get an I/O backend.
    
    Args:
        backend_type: Type of backend ('lmdb', 'disk', 'rocksdb')
        db_path: Path to database/directory
        mode: 'read' or 'write'
        **kwargs: Additional arguments for the backend
        
    Returns:
        I/O backend instance
        
    Example:
        >>> reader = get_io_backend('lmdb', '/path/to/db', mode='read')
        >>> writer = get_io_backend('disk', '/path/to/dir', mode='write')
    """
    if backend_type == "lmdb":
        if mode == "read":
            return LMDBRead(db_path, **kwargs)
        else:
            return LMDBWrite(db_path, **kwargs)
    elif backend_type == "disk":
        if mode == "read":
            return DiskRead(db_path)
        else:
            return DiskWrite(db_path, **kwargs)
    elif backend_type == "rocksdb":
        if mode == "read":
            return RocksDBRead(db_path)
        else:
            return RocksDBWrite(db_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Must be 'lmdb', 'disk', or 'rocksdb'")
