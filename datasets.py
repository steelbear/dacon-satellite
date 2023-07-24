import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.augmentations.transforms import RandomBrightnessContrast
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
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        #plt.imshow(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            # 크롭 들어갈 부분
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


datasets = []
dataloaders = []

for i in range(16):
    x = i % 4
    y = i // 4

    def make_transform(x, y):
        return A.Compose(
            [
                # 1024x1024 사진을 224x224로 축소
                A.Crop(x_min=x * 224, y_min=y * 224, x_max=(x + 1) * 224, y_max=(y + 1) * 224),
                # Normalize 를 뺴서 원본 데이터 보기 실행시 주석 빼기
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=.3, p=.5),
                ToTensorV2()
            ]
        )
    
    transform = make_transform(x, y)

    datasets.append(SatelliteDataset(csv_file='./train.csv', transform=transform))
    dataloaders.append(DataLoader(datasets[i], batch_size=4, shuffle=True, num_workers=1))


_transform = A.Compose(
    [
        #A.CenterCrop(224, 224),  # 1024x1024 사진을 224x224로 축소
        #A.Normalize(),
        RandomBrightnessContrast(p=0.7),
        ToTensorV2()
    ]
)

train_dataset = SatelliteDataset(csv_file='./cropped_train.csv', transform=_transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

test_dataset = SatelliteDataset(csv_file='./test.csv', transform=_transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


if __name__ == '__main__':
    for i in range(16):
        image, mask = datasets[0][i + 20]
        ax = plt.subplot(4, 4, i + 1)
        ax.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()