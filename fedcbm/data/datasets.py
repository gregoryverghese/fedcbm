"""PyTorch datasets for MINOTAUR."""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Union, Optional

# TODO: Update import after removing old cem_mil package
from minotaur.data.io.lmdb import LMDBRead


class TileDataset(Dataset):
    """PyTorch Dataset for loading tile embeddings from database.
    
    Supports multiple task types: binary classification, multiclass, continuous, and Cox regression.
    """
    
    def __init__(
        self,
        dataset: pd.DataFrame,
        db_path: str,
        target: str,
        cpt_ids: List[str],
        bag_num: int,
        task_type: str = 'binary'
    ):
        """
        Initialize TileDataset.
        
        Args:
            dataset: DataFrame with patient data (ID, target, concepts, etc.)
            db_path: Path to database directory containing feature files
            target: Name of target column in dataset
            cpt_ids: List of concept IDs (column names in dataset)
            bag_num: Number of tiles to sample per bag
            task_type: Task type ('binary', 'multiclass', 'continuous', or 'cox')
        """
        self.dataset = dataset
        self.db_path = db_path
        self.db_paths = glob.glob(os.path.join(db_path, 'features', '*'))
        
        # Filter database paths to match dataset IDs
        self.ws_dbs = [
            p
            for p in self.db_paths
            for db in list(dataset['ID'])
            if db == os.path.basename(p)[:-4]
        ][:100]
        
        self.tgt = target
        self.cpt_ids = cpt_ids
        self.bag_num = bag_num
        self.task_type = task_type
        self.lmdb_read = None
        self.multiple_bags = False
        self.bags = self._generate_bags()

    def _get_concepts(self, ws_id: str) -> List[float]:
        """Get concept values for a given WSI ID."""
        cpts = []
        for c in self.cpt_ids:
            cpts.append(
                self.dataset.loc[
                    self.dataset['ID'] == ws_id, c
                ].iloc[0]
            )
        return cpts

    def _get_target(self, ws_id: str) -> float:
        """Get target value for a given WSI ID."""
        return self.dataset.loc[
            self.dataset['ID'] == ws_id, self.tgt
        ].iloc[0]

    def _get_survival(self, ws_id: str) -> Tuple[int, float]:
        """
        Get survival data for Cox regression.
        
        Returns:
            Tuple of (event, survtime) where:
            - event: 1 if event observed, 0 if censored
            - survtime: time to event or time to last follow-up
            
        Raises:
            ValueError: If required columns are missing or invalid
        """
        row = self.dataset.loc[self.dataset['ID'] == ws_id].iloc[0]

        event = row['event']
        survtime = row['time']

        if pd.isna(event):
            raise ValueError(f"Missing event value for {ws_id}")
        if pd.isna(survtime) or survtime <= 0:
            raise ValueError(f"Invalid survival time for {ws_id}: {survtime}")

        return int(event), float(survtime)

    def _sample_tiles(self, keys: List) -> torch.Tensor:
        """Sample random tile indices."""
        return torch.randint(0, len(keys), (self.bag_num,))

    def _generate_bags(self) -> List[Tuple]:
        """Generate bags of tiles for each WSI."""
        bags = []
        for i, db_pth in enumerate(self.ws_dbs):
            ws_id = os.path.basename(db_pth)[:-4]
            self._init_db(db_pth)
            keys = self.lmdb_read.get_keys()
            
            if len(keys) == 0:
                continue
            
            random.shuffle(keys)
            num_bags = 1
            
            for j in range(num_bags):
                n = self.bag_num if len(keys) > self.bag_num else len(keys)
                indices = self._sample_tiles(keys)
                bag = np.array(keys)[indices.numpy()]

                if self.task_type == 'cox':
                    event, survtime = self._get_survival(ws_id)
                    cpts = self._get_concepts(ws_id)
                    bags.append((db_pth, bag, (event, survtime), cpts))
                else:
                    target = self._get_target(ws_id)
                    cpts = self._get_concepts(ws_id)
                    bags.append((db_pth, bag, target, cpts))
        
        return bags

    def _init_db(self, db: str):
        """Initialize LMDB reader for a database."""
        self.lmdb_read = LMDBRead(db)

    def _buildbag(self, id_: str, bag: np.ndarray) -> torch.Tensor:
        """Build a bag tensor from tile keys."""
        self._init_db(id_)
        bag_mat = []
        for k in bag:
            ndarray = self.lmdb_read.read_ndarray(k)
            feature_vec = ndarray.get_ndarray()
            feature_vec = torch.Tensor(np.array(feature_vec))
            feature_vec = torch.squeeze(feature_vec, 0)
            bag_mat.append(feature_vec)
        bag_mat = torch.stack(bag_mat, 0)
        bag_mat = torch.squeeze(bag_mat)
        return bag_mat

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[float, Tuple], torch.Tensor, str]:
        """
        Get a sample from the dataset.
        
        Returns:
            For Cox regression: (bag, (event, survtime), cpts, wsi_id)
            For other tasks: (bag, target, cpts, wsi_id)
        """
        wsi_id, bag, target, cpts = self.bags[index]
        bag = self._buildbag(wsi_id, bag)
        cpts = torch.tensor(cpts, dtype=torch.float32)

        if self.task_type == 'cox':
            event, survtime = target
            event = torch.tensor(event, dtype=torch.float32)
            survtime = torch.tensor(survtime, dtype=torch.float32)
            return bag, (event, survtime), cpts, wsi_id
        else:
            target = torch.tensor(target, dtype=torch.float32)
            return bag, target, cpts, wsi_id

    def __len__(self) -> int:
        """Get the number of bags in the dataset."""
        return len(self.bags)


class MultinominalSampler(torch.utils.data.Sampler):
    """Sampler for weighted sampling from a dataset.
    
    Uses multinomial sampling with optional replacement.
    """
    
    def __init__(
        self,
        weights: List[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None
    ):
        """
        Initialize MultinominalSampler.
        
        Args:
            weights: List of weights (does not need to be normalized)
            num_samples: Number of samples to draw at each iteration
            replacement: Whether to sample with replacement (default: True)
            generator: Random number generator (default: None)
        """
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        """Generate sample indices."""
        rand_tensor = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=self.generator
        )
        return iter(list(rand_tensor))

    def __len__(self) -> int:
        """Get the number of samples."""
        return self.num_samples


