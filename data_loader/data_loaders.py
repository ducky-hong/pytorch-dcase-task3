import os
import glob
import itertools
import numpy as np
import torch
from base import BaseDataLoader
from torch.utils.data import Dataset, DataLoader
        
class FeatureNpyDataset(Dataset):
    def __init__(self, root_dir, datasets, transform=None):
        self.data = list(itertools.chain(*[glob.glob(os.path.join(root_dir, dataset_id, '*/*.npy')) for dataset_id in datasets]))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filepath = self.data[0]
        has_bird = int(filepath.split('/')[-2])
        feature = np.load(filepath)

        if self.transform:
            feature = self.transform(feature)

        return feature, has_bird

class BADDataLoader(BaseDataLoader):
    def __init__(self, data_dir, sample_rate, fold, batch_size, shuffle, validation_split, num_workers):
        self.data_dir = os.path.join(data_dir, str(sample_rate))
        self.batch_size = batch_size

        train_sets = os.listdir(self.data_dir)
        validation_sets = [train_sets.pop(fold)]
        
        transform = lambda a: torch.from_numpy(np.expand_dims(a, axis=0)).float()
        self.dataset = FeatureNpyDataset(self.data_dir, train_sets, transform=transform)
        self.validation_dataset = FeatureNpyDataset(self.data_dir, validation_sets, transform=transform)
        super(BADDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def split_validation(self):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs['dataset'] = self.validation_dataset
        return DataLoader(**init_kwargs)
