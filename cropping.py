from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv

from utils import rle_decode, rle_encode

if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')

    subimg_tag = 'ABCDEFGHIJKLMNOP'

    csvfile = open('cropped_train.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['img_id', 'img_path', 'mast_rle'])
    
    for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        big_img_id = row['img_id']
        big_img_path = row['img_path']
        big_img = Image.open(big_img_path)
        big_mask = Image.fromarray(rle_decode(row['mask_rle'], (1024, 1024)))

        for i in range(16):
            x = i % 4
            y = i // 4

            img_id = big_img_id + subimg_tag[i]
            img_path = './cropped_train_img/' + img_id + '.png'
            img = big_img.crop((x * 224, y * 224, (x + 1) * 224, (y + 1) * 224))
            mask = big_mask.crop((x * 224, y * 224, (x + 1) * 224, (y + 1) * 224))
            mask_rle = rle_encode(np.array(mask))

            if len(mask_rle) == 0:
                mask_rle = "-1"

            writer.writerow([img_id, img_path, mask_rle])

            img.save(img_path)
    
    csvfile.close()