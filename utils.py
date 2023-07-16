import torch
import numpy as np
from typing import List, Union
from joblib import Parallel, delayed

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def dice_score_torch(prediction, ground_truth, threshold=0.35, smooth=1e-7):
    '''
    prediction: 예측값이 담긴 tensor (batch_size, 1, height, width)
    ground_truth: 정답이 담긴 tensor (batch_size, 1, height, width)

    결과: tensor로 계산한 각 이미지 dice coefficient의 합
    '''
    prediction = torch.nn.functional.sigmoid(prediction)
    prediction = torch.where(prediction > threshold, 1, 0)
    intersection = torch.sum(prediction * ground_truth, [1, 2, 3])
    return ((2.0 * intersection + smooth) / (torch.sum(prediction, [1, 2, 3]) + torch.sum(ground_truth, [1, 2, 3]) + smooth)).sum()


# 한 이미지에 대한 dice coefficient 계산
def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    '''
    prediction: 모델이 예측한 건물 이진 마스크
              : 반드시 실수값이 아닌 이진값이여야 함

    ground_truth: 정답 이진 마스크

    결과: intersection은 두 마스크가 겹치는 넓이로 두고 계산한 dice coefficient
    '''
    intersection = np.sum(prediction * ground_truth) # 서로 element-wise 곱을 하면 둘이 1인 영역만 1이고 나머지는 0
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)


# 모든 이미지에 대한 dice coefficient 계산
def calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)) -> List[float]:
    '''
    ground_truth_df: 정답이 담겨있는 dataframe

    prediction_df: 모델의 예측값이 담겨있는 dataframe

    결과: 모든 이미지 마스크의 dice coefficient 평균값
    '''

    # 두 dataframe의 image_id가 겹치는지 확인하고 겹치는 데이터만 prediction_df에 남겨둔다
    prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]
    prediction_df.index = range(prediction_df.shape[0])


    # rle 인코딩된 마스크
    pred_mask_rle = prediction_df.iloc[:, 1]
    gt_mask_rle = ground_truth_df.iloc[:, 1]


    # rle 인코딩된 결과값을 디코딩하고 dice coefficient를 구하는 함수
    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)

        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # 실제로 건물이 없고 모델의 예측도 건물이 없다고 나오면 계산에서 제외


    # 멀티 쓰레드로 모든 이미지 계산
    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )


    dice_scores = [score for score in dice_scores if score is not None]  # None 버리기


    return np.mean(dice_scores)