from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os
from torch.utils.data import Dataset

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description

    def __init__(self, data, mode):
        super(ChallengeDataset, self).__init__()
        self.data = data
        self.mode = mode
        self.indices = np.asarray(data.index)
        if self.mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_frame = self.data.to_numpy()  # Convert data to a numpy array for ease of reading
        labels = data_frame[index][1:]  # Get value of column 'inactive', 'crack'
        # print(labels, type(labels))
        labels = np.array(labels, dtype='float32')
        path = data_frame[index][0]  # Get image file
        img = imread(path)
        img = gray2rgb(img)
        return self._transform(img), torch.tensor(labels)
