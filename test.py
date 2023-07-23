import torch
from torch.multiprocessing import freeze_support
import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import test_dataloader
from utils import rle_encode


if __name__ == '__main__':
    freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('testing a model on', device)

    # model 초기화
    model = torch.load('./checkpoints/UNet_0720-214328.pth')

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            
            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)
    
    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result

    submit.to_csv('./submit.csv', index=False)