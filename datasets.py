import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import rle_decode

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as T



class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=True, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        if self.infer:
            return len(self.data)
        else:
            return len(self.data) * 16

    def __getitem__(self, idx):
        if self.infer:
            img_path = self.data.iloc[idx, 1]
            img_id = self.data.iloc[idx, 0]

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = _transform(image=image)['image']

            return image, img_id

        img_path = self.data.iloc[idx // 16, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rle = self.data.iloc[idx // 16, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        plt.imshow(mask)

        if self.transform:
            #크롭을 땡켜야 하는 부분
            i = idx % 16
            x = i % 4
            y = i // 4
            transform = make_transform(x, y)
            augmented = transform(image=image, mask=mask)

            image = augmented['image']
            mask = augmented['mask']

            #transform = T.ToPILImage()
            #img = transform(image)
            #img.show()

        return image, mask


def make_transform(x, y):
    return A.Compose(
        [
            A.Crop (x_min=x*224, y_min=y*224, x_max=(x+1)*224, y_max=(y+1)*224),  # 1024x1024 사진을 224x224로 축소
            A.Normalize(),
            ToTensorV2()
        ]
    )


_transform = A.Compose(
    [
        #A.Crop (x_min=0, y_min=0, x_max=224, y_max=224),  # 1024x1024 사진을 224x224로 축소
        A.Normalize(),
        ToTensorV2()
    ]
)

dataset = SatelliteDataset(csv_file='./train.csv')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

test_dataset = SatelliteDataset(csv_file='./test.csv', infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)



if __name__ == '__main__':
    image, mask = dataset[0]
    image = np.floor(image * 255)

    #plt.imshow(np.transpose(image, (1, 2, 0)))
    #plt.show()
    #plt.imshow(mask)
    #plt.show()