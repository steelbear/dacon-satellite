import sys
import torch
from torch.multiprocessing import freeze_support
import torchvision
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

from time import localtime, strftime
from tqdm import tqdm
from models import UNet
from datasets import test_dataloader



if __name__ == '__main__':
    freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('testing a model on', device)

    # model 초기화
    model = torch.load('./checkpoints/UNet_0718-102345.pth')

    # loss function과 optimizer 정의
    sigmoid = torch.nn.Sigmoid()

    loss = 0
    with torch.no_grad():
        model.eval()
        for images, img_id in tqdm(test_dataloader):
            images = images.float().to(device)
            
            outputs = model(images)
            for i in range(16):
                mask = torch.sigmoid(outputs[i]).cpu().numpy()
                mask = np.squeeze(mask, axis=0)
                mask = (mask > 0.35).astype(np.uint8)

                plt.imshow(mask)
                plt.savefig('./results/'+ img_id[i] +'.png')