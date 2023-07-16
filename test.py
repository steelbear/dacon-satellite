import torch
from torch.multiprocessing import freeze_support
import torchvision
from torchvision.utils import save_image

from time import localtime, strftime
from tqdm import tqdm
from models import UNet
from datasets import test_dataloader



if __name__ == '__main__':
    freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training a model on', device)

    # 하이퍼 파라미터
    EPOCHS = 3
    LEARNING_RATE = 0.001

    # model 초기화
    model = UNet().to(device)

    model = torch.load('./checkpoints/UNet_0710-062508.pth')

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    i = 0
    loss = 0
    for images,info in tqdm(test_dataloader):
        images = images.float().to(device)
        num_str = str(i)
        outputs = model(images)

        info = str(info)
        save_image(outputs, './results/'+info+'.png')



        #loss += loss.item()