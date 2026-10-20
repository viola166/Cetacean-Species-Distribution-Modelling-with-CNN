import torch
import numpy as np
from data.prepareGridData import make_patches

class OccurrenceDataset(torch.utils.data.Dataset):
    def __init__(self, Y, W, meta, norm_dss, mask_dss=None, augment=False):                  # already normalized channel datasets
        X = [make_patches(norm_dss, mask_dss, r.latitude, r.longitude, r.time)
             for r in meta.itertuples()]
        self.X = torch.from_numpy(np.stack(X))     # (N, 2C, 13, 13)
        self.Y = torch.from_numpy(Y)               # (N, S)
        self.W = torch.from_numpy(W)               # (N, S)
        self.meta = meta                           # kept for debugging and map plots
        self.augment = augment

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, n):
        x = self.X[n]
        if self.augment:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])            # horizontal flip
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])            # vertical flip
            if torch.rand(1) < 0.5:
                k = int(torch.randint(1, 4, (1,)))     # up to three 90° rotations
                x = torch.rot90(x, k, dims=[1, 2])

        return x, self.Y[n], self.W[n]


    
