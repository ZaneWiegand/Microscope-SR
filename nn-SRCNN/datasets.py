# %%
import h5py
import numpy as np
from torch.utils.data import Dataset
# %%


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
