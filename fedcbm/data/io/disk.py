"""Disk-based I/O backend for MINOTAUR."""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple


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


class DiskWrite:
    """Disk writer for tile embeddings."""
    
    def __init__(self, db_path: Union[str, Path], write_frequency: int = 10):
        """
        Initialize disk writer.
        
        Args:
            db_path: Path to directory for storing tiles
            write_frequency: Progress update frequency
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.write_frequency = write_frequency

    def __repr__(self) -> str:
        return f'DiskWrite(path: {self.db_path})'

    def _print_progress(self, i: int, total: int):
        """Print write progress."""
        complete = float(i) / total
        print(f'\r- Progress: {complete:.1%}', end='\r')

    def write(self, x: List[np.ndarray], keys: List[str], attentions: List[float],
              score_0s: List[float], score_1s: List[float]):
        """
        Write tiles with attention and scores to disk (pickle format).
        
        Args:
            x: List of numpy arrays (tile embeddings)
            keys: List of string keys
            attentions: List of attention values
            score_0s: List of score 0 values
            score_1s: List of score 1 values
        """
        for i in range(len(attentions)):
            key = f"{keys[i]}"
            value = Tile(x[i], attentions[i], score_0s[i], score_1s[i])
            with open(self.db_path / f"{key}.pkl", "wb") as f:
                pickle.dump(value, f)
            if i % self.write_frequency == 0:
                self._print_progress(i, len(attentions))

    def write_from_parser(self, parser):
        """
        Write tiles from a parser generator (numpy format).
        
        Args:
            parser: Generator yielding (coordinate, tile) tuples
        """
        tile_buffer = []
        meta_buffer = []

        for i, (p, tile) in enumerate(parser):
            name = str(p[1]) + '_' + str(p[0])
            tile_path = self.db_path / f"{name}.npy"
            meta_path = self.db_path / f"{name}_meta.pkl"

            tile_buffer.append((tile_path, tile))
            meta_buffer.append((meta_path, {'size': tile.shape, 'dtype': tile.dtype}))

            if (i + 1) % self.write_frequency == 0:
                self._write_buffer(tile_buffer, meta_buffer)
                tile_buffer = []
                meta_buffer = []

        if tile_buffer:
            self._write_buffer(tile_buffer, meta_buffer)

        print("\nFinished writing tiles to disk.")

    def _write_buffer(self, tile_buffer: List[Tuple[Path, np.ndarray]], 
                     meta_buffer: List[Tuple[Path, dict]]):
        """
        Write buffer contents to disk.
        
        Args:
            tile_buffer: List of (path, tile) tuples
            meta_buffer: List of (path, metadata) tuples
        """
        for tile_path, tile in tile_buffer:
            np.save(tile_path, tile)

        for meta_path, metadata in meta_buffer:
            with open(meta_path, 'wb') as meta_file:
                pickle.dump(metadata, meta_file)

    def close(self):
        """Close the writer (no-op for disk I/O)."""
        pass


class DiskRead:
    """Disk reader for tile embeddings."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize disk reader.
        
        Args:
            db_path: Path to directory containing tiles
        """
        self.db_path = Path(db_path)

    def __repr__(self) -> str:
        return f'DiskRead(path: {self.db_path})'

    @property
    def num_keys(self) -> int:
        """Get the number of keys in the database."""
        return len([f for f in self.db_path.glob('*.npy')])

    def get_keys(self) -> List[str]:
        """Get all keys from the database."""
        # Prefer .npy files (from write_from_parser), fall back to .pkl
        npy_keys = [f.stem for f in self.db_path.glob("*.npy")]
        if npy_keys:
            return npy_keys
        return [f.stem for f in self.db_path.glob("*.pkl")]

    def read_size(self, key: str) -> tuple:
        """
        Read the size of an array from disk.
        
        Args:
            key: Key to read
            
        Returns:
            Size tuple
        """
        pkl_path = self.db_path / f"{key}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                tile = pickle.load(f)
            return tile.size
        else:
            # For .npy files, load and check shape
            npy_path = self.db_path / f"{key}.npy"
            if npy_path.exists():
                tile = np.load(npy_path)
                return tile.shape
        raise FileNotFoundError(f"No file found for key: {key}")

    def read_ndarray(self, key: str) -> NpyObject:
        """
        Read a numpy array object from disk.
        
        Args:
            key: Key to read
            
        Returns:
            NpyObject or Tile object
        """
        # Try .npy file first (from write_from_parser)
        npy_path = self.db_path / f"{key}.npy"
        if npy_path.exists():
            tile = np.load(npy_path)
            return NpyObject(tile)
        
        # Fall back to .pkl file (from write method)
        pkl_path = self.db_path / f"{key}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                tile = pickle.load(f)
            return tile
        
        raise FileNotFoundError(f"No file found for key: {key}")

    def read_attentions(self) -> List[float]:
        """Read all attention values from disk."""
        keys = self.get_keys()
        attentions = []
        for k in keys:
            tile = self.read_ndarray(k)
            if hasattr(tile, 'get_attention'):
                attentions.append(tile.get_attention())
        return attentions

    def read_scores(self) -> Tuple[List, List]:
        """
        Read all scores from disk.
        
        Returns:
            Tuple of (score_0_list, score_1_list)
        """
        keys = self.get_keys()
        scores_0, scores_1 = [], []
        for k in keys:
            tile = self.read_ndarray(k)
            if hasattr(tile, 'get_score_0'):
                scores_0.append(tile.get_score_0())
                scores_1.append(tile.get_score_1())
        return scores_0, scores_1

    def read_embeddings(self) -> List[np.ndarray]:
        """Read all embeddings from disk."""
        keys = self.get_keys()
        embeddings = []
        for k in keys:
            tile = self.read_ndarray(k)
            embeddings.append(tile.get_ndarray())
        return embeddings


