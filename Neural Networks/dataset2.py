import h5py
import torch
from itertools import accumulate
import torch.utils.data
import zarr


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_file_list, is_trainning=True):
        siz_list = []
        self.train_label = 'train' if is_trainning is True else 'val'
        # self.zarr_file_list = []
        for zarr_file in zarr_file_list:
            store = zarr.LMDBStore(zarr_file)
            h5_ds = zarr.group(store=store)
            # self.zarr_file_list.append(h5_ds[self.train_label])
            siz_list.append(h5_ds[self.train_label]['features'].shape[0])
            store.close()
        self.size_list=[0]
        self.size_list.extend(list(accumulate(siz_list)))
        self.zarr_file_list = zarr_file_list
        # print(zarr_file_list)

    def __getitem__(self, idx):
        new_idx = 0
        feature_id = 0
        for i, siz in enumerate(self.size_list):
            if idx < siz:
                new_idx = idx - self.size_list[i-1]
                feature_id = i-1
                break
        # print(feature_id)
        with zarr.LMDBStore(self.zarr_file_list[feature_id]) as store:
            # store = zarr.LMDBStore(self.zarr_file_list[feature_id])
            ds = zarr.group(store=store)[self.train_label]
            feature = torch.tensor(ds['features'][new_idx, :])
            label = torch.tensor(ds['labels'][new_idx], dtype=torch.long)

        return feature, label

    def __len__(self):
        return self.size_list[-1]
