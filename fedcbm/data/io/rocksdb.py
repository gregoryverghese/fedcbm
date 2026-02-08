"""RocksDB I/O backend for MINOTAUR."""
import pickle
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

try:
    import rocksdb
except ImportError:
    rocksdb = None
    import warnings
    warnings.warn("rocksdb not available. Install with: pip install python-rocksdb")


class NpyObject:
    """Object wrapper for numpy arrays for serialization."""
    
    def __init__(self, ndarray: np.ndarray):
        self.ndarray = ndarray.tobytes()
        self.size = ndarray.shape
        self.dtype = ndarray.dtype

    def get_ndarray(self) -> np.ndarray:
        """Get the numpy array from the object."""
        ndarray = np.frombuffer(self.ndarray, dtype=self.dtype)
        return ndarray.reshape(self.size)


class RocksDBWrite:
    """RocksDB writer for tile embeddings."""
    
    def __init__(self, db_path: Union[str, Path], write_frequency: int = 10):
        """
        Initialize RocksDB writer.
        
        Args:
            db_path: Path to RocksDB database
            write_frequency: Commit frequency for writes
        """
        if rocksdb is None:
            raise ImportError("rocksdb not available. Install with: pip install python-rocksdb")
        
        self.db_path = Path(db_path)
        print(f"DB Path: {self.db_path}")

        options = rocksdb.Options()
        options.create_if_missing = True
        options.write_buffer_size = 64 * 1024 * 1024  # 64MB
        options.max_write_buffer_number = 3
        options.target_file_size_base = 64 * 1024 * 1024
        self.db = rocksdb.DB(str(self.db_path), options)
        self.write_frequency = write_frequency

    def __repr__(self) -> str:
        return f'RocksDBWrite(path: {self.db_path})'

    def _print_progress(self, i: int, total: int):
        """Print write progress."""
        complete = float(i) / total
        print(f'\r- Progress: {complete:.1%}', end='\r')

    def write(self, parser):
        """
        Write tiles from a parser generator to RocksDB.
        
        Args:
            parser: Generator yielding (coordinate, tile) tuples
        """
        batch = rocksdb.WriteBatch()
        total = sum(1 for _ in parser)

        for i, (p, tile) in enumerate(parser):
            name = str(p[1]) + '_' + str(p[0])
            key = f"{name}".encode("ascii")
            value = NpyObject(tile)
            batch.put(key, pickle.dumps(value))

            if i % self.write_frequency == 0:
                self.db.write(batch)
                batch.clear()
                self._print_progress(i, total)

        if not batch.empty():
            self.db.write(batch)

    def write_image(self, image: np.ndarray, name: str):
        """
        Write a single image to RocksDB.
        
        Args:
            image: Image array
            name: Key name
        """
        key = f"{name}".encode('ascii')
        value = NpyObject(image)
        self.db.put(key, pickle.dumps(value))

    def close(self):
        """Close the RocksDB database."""
        del self.db


class RocksDBRead:
    """RocksDB reader for tile embeddings."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize RocksDB reader.
        
        Args:
            db_path: Path to RocksDB database
        """
        if rocksdb is None:
            raise ImportError("rocksdb not available. Install with: pip install python-rocksdb")
        
        self.db_path = Path(db_path)
        options = rocksdb.Options()
        options.create_if_missing = False
        self.db = rocksdb.DB(str(self.db_path), options, read_only=True)

    @property
    def num_keys(self) -> int:
        """Get the number of keys in the database."""
        it = self.db.iterkeys()
        it.seek_to_first()
        return sum(1 for _ in it)

    def __repr__(self) -> str:
        return f'RocksDBRead(path: {self.db_path})'

    def get_keys(self) -> List[str]:
        """Get all keys from the database."""
        keys = []
        it = self.db.iterkeys()
        it.seek_to_first()
        for key in it:
            keys.append(key.decode('ascii'))
        return keys

    def read_image(self, key: str) -> Optional[np.ndarray]:
        """
        Read an image array from RocksDB.
        
        Args:
            key: Key to read
            
        Returns:
            Numpy array or None if not found
        """
        value = self.db.get(key.encode('ascii'))
        if value:
            image = pickle.loads(value)
            return image.get_ndarray()
        return None


