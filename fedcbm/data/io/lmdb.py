"""LMDB I/O backend for MINOTAUR."""
import os
import pickle
import sys
import numpy as np
import lmdb
from typing import List, Optional, Union
from pathlib import Path

# Fix for pickle module import issue
# This ensures that when unpickling objects that reference 'lmdb_io',
# Python can find the module
if 'lmdb_io' not in sys.modules:
    sys.modules['lmdb_io'] = sys.modules[__name__]


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


class Tile(NpyObject):
    """Tile object with embeddings, attention, and scores."""
    
    def __init__(self, ndarray: np.ndarray, attention: float, score_0: float, score_1: float):
        super().__init__(ndarray)
        self.attention = np.float64(attention).tobytes()
        self.score_0 = np.float64(score_0).tobytes()
        self.score_1 = np.float64(score_1).tobytes()

    def get_attention(self) -> float:
        """Get attention value."""
        attention = np.frombuffer(self.attention, dtype=np.float64)[0]
        return attention

    def get_score_0(self) -> np.ndarray:
        """Get score 0."""
        score_0 = np.frombuffer(self.score_0)
        return score_0

    def get_score_1(self) -> np.ndarray:
        """Get score 1."""
        score_1 = np.frombuffer(self.score_1)
        return score_1


class LMDBWrite:
    """LMDB writer for tile embeddings."""
    
    def __init__(self, db_path: Union[str, Path], map_size: int = 1073741824, write_frequency: int = 10):
        """
        Initialize LMDB writer.
        
        Args:
            db_path: Path to LMDB database
            map_size: Maximum database size in bytes (default: 1GB)
            write_frequency: Commit frequency for writes
        """
        self.db_path = Path(db_path)
        self.map_size = map_size
        self.env = lmdb.open(str(self.db_path), map_size=self.map_size, writemap=True)
        self.write_frequency = write_frequency

    def __repr__(self) -> str:
        return f'LMDBWrite(size: {self.map_size}, path: {self.db_path})'

    def _print_progress(self, i: int, total: int):
        """Print write progress."""
        complete = float(i) / total
        print(f'\r- Progress: {complete:.1%}', end='\r')

    def write(self, x: List[np.ndarray], keys: List[str], attentions: List[float], 
              score_0s: List[float], score_1s: List[float]):
        """
        Write tiles with attention and scores to LMDB.
        
        Args:
            x: List of numpy arrays (tile embeddings)
            keys: List of string keys
            attentions: List of attention values
            score_0s: List of score 0 values
            score_1s: List of score 1 values
        """
        txn = self.env.begin(write=True)
        for i in range(len(attentions)):
            key = f"{keys[i]}"
            value = Tile(x[i], attentions[i], score_0s[i], score_1s[i])
            txn.put(key.encode("ascii"), pickle.dumps(value))
            if i % self.write_frequency == 0:
                txn.commit()
                txn = self.env.begin(write=True)
        txn.commit()
        self.env.close()

    def write_from_parser(self, parser):
        """
        Write tiles from a parser generator (for compatibility with tiler version).
        
        Args:
            parser: Generator yielding (coordinate, tile) tuples
        """
        print("Beginning writing to db ...")
        txn = self.env.begin(write=True)
        for i, (p, tile) in enumerate(parser):
            name = str(p[1]) + '_' + str(p[0])
            key = f"{name}"
            value = NpyObject(tile)
            txn.put(key.encode("ascii"), pickle.dumps(value))
            if i % self.write_frequency == 0:
                txn.commit()
                txn = self.env.begin(write=True)
        txn.commit()
        self.env.close()
        print("Writing to db done")

    def write_image(self, image: np.ndarray, name: str):
        """
        Write a single image to LMDB.
        
        Args:
            image: Image array
            name: Key name
        """
        txn = self.env.begin(write=True)
        value = NpyObject(image)
        b = pickle.dumps(value)
        key = f"{name}"
        txn.put(key.encode('ascii'), b)
        txn.commit()

    def close(self):
        """Close the LMDB environment."""
        self.env.close()


class LMDBRead:
    """LMDB reader for tile embeddings."""
    
    def __init__(self, db_path: Union[str, Path], image_size: Optional[tuple] = None):
        """
        Initialize LMDB reader.
        
        Args:
            db_path: Path to LMDB database
            image_size: Optional image size (not used, kept for compatibility)
        """
        self.db_path = Path(db_path)
        self.env = lmdb.open(str(self.db_path), readonly=True, lock=False)

    @property
    def num_keys(self) -> int:
        """Get the number of keys in the database."""
        with self.env.begin() as txn:
            length = txn.stat()['entries']
        return length

    def __repr__(self) -> str:
        return f'LMDBRead(path: {self.db_path})'

    def get_keys(self) -> List[bytes]:
        """Get all keys from the database."""
        txn = self.env.begin()
        keys = [k for k, _ in txn.cursor()]
        return keys

    def read_size(self, key: bytes) -> tuple:
        """
        Read the size of an array from the database.
        
        Args:
            key: Key to read
            
        Returns:
            Size tuple
        """
        txn = self.env.begin()
        data = txn.get(key)
        ndarray = pickle.loads(data)
        return ndarray.size

    def read_ndarray(self, key: bytes) -> NpyObject:
        """
        Read a numpy array object from the database.
        
        Args:
            key: Key to read
            
        Returns:
            NpyObject or Tile object
        """
        txn = self.env.begin()
        data = txn.get(key)
        ndarray = pickle.loads(data)
        return ndarray

    def read_image(self, key: bytes) -> np.ndarray:
        """
        Read an image array from the database (for compatibility).
        
        Args:
            key: Key to read
            
        Returns:
            Numpy array
        """
        txn = self.env.begin()
        data = txn.get(key)
        image = pickle.loads(data)
        image = image.get_ndarray()
        return image

    def read_attentions(self) -> List[float]:
        """Read all attention values from the database."""
        keys = self.get_keys()
        attentions = []
        for k in keys:
            ndarray = self.read_ndarray(k)
            if hasattr(ndarray, 'get_attention'):
                attentions.append(ndarray.get_attention())
        return attentions

    def read_scores(self) -> tuple:
        """
        Read all scores from the database.
        
        Returns:
            Tuple of (score_0_list, score_1_list)
        """
        keys = self.get_keys()
        scores_0, scores_1 = [], []
        for k in keys:
            ndarray = self.read_ndarray(k)
            if hasattr(ndarray, 'get_score_0'):
                scores_0.append(ndarray.get_score_0())
                scores_1.append(ndarray.get_score_1())
        return scores_0, scores_1

    def read_embeddings(self) -> List[np.ndarray]:
        """Read all embeddings from the database."""
        keys = self.get_keys()
        embeddings = []
        for k in keys:
            ndarray = self.read_ndarray(k)
            embeddings.append(ndarray.get_ndarray())
        return embeddings


