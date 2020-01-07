import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split

class RecommenderDataset():
    def __init__(self, filename, implicit=False, rescale=False,
                 train_size=.8, validation_size=.1, implicit_fill_val=0,
                 loader_kwargs={}, data_kwargs={}):
        self.orig_df = self.load_data(filename, **data_kwargs)
        self.ratings_df = self.orig_df.copy()
        self.remap_ids()
        if rescale:
            self.rescale()
        if implicit:
            self.convert_to_implicit(implicit_fill_val)

        self.dataset = TensorDataset(torch.from_numpy(self.ratings_df.userId_remap.values),
                                     torch.from_numpy(self.ratings_df.itemId_remap.values),
                                     torch.from_numpy(self.ratings_df.rating.values).float())

        train_len = int(len(self.dataset)*train_size)
        val_len = int(len(self.dataset)*validation_size)
        test_len = len(self.dataset) - train_len - val_len
        train_dataset, val_dataset, test_dataset = random_split(self.dataset,
                                                                [train_len,
                                                                val_len,
                                                                test_len])
        self.datasets = {'train': train_dataset,
                         'val': val_dataset,
                         'test': test_dataset}
        self.loaders = self.create_loaders(**loader_kwargs)

    def convert_to_implicit(self, fill_fal=0):
        rating_mat = data.get_rating_mat()
        rating_mat.fillna(fill_val, inplace=True)
        return pd.melt(rating_mat.reset_index(), id_vars='userId')

    def get_stats(self):
        stats = {}
        stats['n_users'] = len(self.ratings_df.userId.unique())
        stats['n_items'] = len(self.ratings_df.itemId.unique())
        total = stats['n_users'] * stats['n_items']
        stats['missing_percentage'] = 1-(self.ratings_df.shape[0]/total)
        return stats

    def create_loaders(self, batch_size=64, **kwargs):
        loaders = {}
        for key, val in self.datasets.items():
            loaders[key] = train_loader = DataLoader(val,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    **kwargs)
        return loaders

    def get_rating_mat(self):
        return pd.pivot(self.ratings_df, 'userId', 'itemId', 'rating')

    def view_data_subset(self, key='train'):
        return self.ratings_df.iloc[self.datasets[key].indices]

    def load_data(self, filename):
        pass


    def remap_ids(self):
        # remap itemIds for embedding
        sorted_ids = np.sort(self.ratings_df.itemId.unique())
        self.itemIdmap = {item:i for item, i in zip(sorted_ids,
                                               range(1,len(sorted_ids)+1))}
        self.ratings_df.loc[:,'itemId_remap'] = [self.itemIdmap[x]
                                            for x in self.ratings_df.itemId]
        # remap userIds for embedding
        sorted_ids = np.sort(self.ratings_df.userId.unique())
        self.userIdmap = {user:i for user, i in zip(sorted_ids,
                                               range(1,len(sorted_ids)+1))}
        self.ratings_df.loc[:,'userId_remap'] = [self.userIdmap[x]
                                            for x in self.ratings_df.userId]

    def rescale(self):
        self.ratings_df.rating = self.ratings_df.rating/self.ratings_df.rating.max()


class MovieLens(RecommenderDataset):
    def load_data(self, filename, **kwargs):
        ratings_df = pd.read_csv(filename, **kwargs)
        ratings_df.rename(columns={'movieId': 'itemId'}, inplace=True)
        return ratings_df
