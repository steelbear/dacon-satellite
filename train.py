import torch
from torch.multiprocessing import freeze_support
from time import strftime
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import UNet
from msnet import MSU_Net
from resunet import build_resunetplusplus
from network import AttU_Net
from datasets import dataloaders
from utils import dice_score_torch, dice_loss

crop_num = 0

if __name__ == '__main__':
    freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print('training a model on', device)


    # 하이퍼 파라미터
    EPOCHS = 10
    LEARNING_RATE = 0.001

    len_dataset = len(dataloaders[0]) * 16 * 4

    # model 초기화
    model = AttU_Net().to(device)

    # 모델 저장 파일 이름
    checkpoint_name = model.__class__.__name__ + '_' + strftime('%m%d-%H%M%S') + '.pth'

    losses = []

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in range(EPOCHS * 16):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        epoch_dice = 0

        i = epoch % 16

        for images, masks in tqdm(dataloaders[i]):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += dice_score_torch(outputs, masks.unsqueeze(1)).item()
        
        mean_epoch_loss = epoch_loss / len_dataset
        losses.append(mean_epoch_loss)

        if (epoch + 1) % 16 == 0:
            torch.save(model, './checkpoints/' + checkpoint_name[:-4] + '_' + str(epoch) + '.pth')
            plt.plot(losses)
            plt.xlabel('epoch')
            plt.ylabel('losses')
            plt.savefig('./train_losses.png')
            plt.close()

        print(f'Epoch {epoch // 16} - {epoch % 16 + 1}, Loss: {mean_epoch_loss}, Dice: {epoch_dice / len_dataset}')

    torch.save(model, './checkpoints/' + checkpoint_name) # 학습된 모델 파일 저장