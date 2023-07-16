import torch
from torch.multiprocessing import freeze_support

from time import localtime, strftime
from tqdm import tqdm
from models import UNet
from datasets import dataloader


import torchvision.transforms as T
from PIL import Image

crop_num = 0

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print('training a model on', device)


    # 하이퍼 파라미터
    EPOCHS = 20
    LEARNING_RATE = 0.003

    # model 초기화
    model = UNet().to(device)

    # 모델 저장 파일 이름
    checkpoint_name = model.__class__.__name__ + '_' + strftime('%m%d-%H%M%S') + '.pth'

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in range(EPOCHS):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        
        for i in range(16):
            for images, masks in tqdm(dataloader[i]):
                images = images.float().to(device)
                masks = masks.float().to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            print(f'Epoch {epoch * 16 + i + 1}, Loss: {epoch_loss / len(dataloader)}')



    torch.save(model, './checkpoints/' + checkpoint_name) # 학습된 모델 파일 저장