import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import rle_decode

import numpy as np
import matplotlib.pyplot as plt


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
                img_id = self.data.iloc[idx, 0]
            return image, img_id

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        #plt.imshow(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            # 크롭 들어갈 부분
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


_transform = A.Compose(
    [
        A.CenterCrop(224,224),  # 1024x1024 사진을 224x224로 축소
        A.Normalize(),
        ToTensorV2()
    ]
)

dataset = SatelliteDataset(csv_file='./train.csv', transform=_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

test_dataset = SatelliteDataset(csv_file='./test.csv', transform=_transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


if __name__ == '__main__':
    image, mask = dataset[0]
    image = np.floor(image * 255)

    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()
    plt.imshow(mask)
    plt.show()