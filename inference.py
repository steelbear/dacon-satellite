import torch
from torch.multiprocessing import freeze_support
import torchvision
from torchvision.utils import save_image

from time import localtime, strftime
import numpy as np
from tqdm import tqdm
from models import UNet
from datasets import test_dataloader
from utils import rle_encode
import pandas as pd
from PIL import Image
import cv2

if __name__ == '__main__':
    freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('testing a model on', device)

    # 하이퍼 파라미터
    EPOCHS = 3
    LEARNING_RATE = 0.001

    # model 초기화
    #model = UNet().to(device)

    model = torch.load('./checkpoints/SegNet_0722-221543191.pth')

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    i = 0
    loss = 0
    j = 0
    result = []
    with torch.no_grad():
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)  # Threshold = 0.35

            info = str(j)

            masks = masks.reshape(224, 224) * 255
            img_2 = Image.fromarray(masks)
            img_2.save('./results/' + info + '_before.png', 'JPEG')

            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            masks = cv2.erode(masks, k)
            masks = cv2.erode(masks, k)
            masks = cv2.dilate(masks, k)
            masks = cv2.dilate(masks, k)


            img_2 = Image.fromarray(masks)
            img_2.save('./results/' + info + '_after.png', 'JPEG')

            #img_2 = Image.fromarray(masks)

            #img_2.save('./results/' + info + '.png', 'JPEG')
            #save_image(outputs, './results/' + info + '.png')
            j = j + 1
            masks = masks.reshape(1, 1, 224, 224) / 255

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

        submit = pd.read_csv('./sample_submission.csv')
        submit['mask_rle'] = result

        submit.to_csv('./submit.csv', index=False)