import numpy as np
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim
import torch

def cal_psnr_ssim_list(real_y_list, pred_y_list, return_total=False):
    total_psnr = np.zeros(len(real_y_list))
    total_ssim = np.zeros(len(real_y_list))
    for idx in range(len(real_y_list)):
        total_psnr[idx] = cal_psnr(real_y_list[idx], pred_y_list[idx], data_range=1)
        total_ssim[idx] = cal_ssim(real_y_list[idx], pred_y_list[idx], data_range=1)
    if return_total:
        return total_psnr.mean(), total_ssim.mean(), total_psnr, total_ssim
    else:
        return total_psnr.mean(), total_ssim.mean()

def cal_dice_score(real, pred, smooth=1e-7, reduction_dim=2):
    batch_size = real.shape[0]
    num_labels = real.shape[1]
    if isinstance(real, torch.Tensor):
        real = real.view(batch_size, num_labels, -1)
        pred = pred.view(batch_size, num_labels, -1)
        intersection = torch.sum(pred * real, dim=reduction_dim)
        cardinality = torch.sum(pred + real, dim=reduction_dim)
    elif isinstance(real, np.ndarray):
        real = real.reshape(batch_size, num_labels, -1)
        pred = pred.reshape(batch_size, num_labels, -1)
        intersection = np.sum(pred * real, axis=reduction_dim)
        cardinality = np.sum(pred + real, axis=reduction_dim)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice_score

def cal_dice_score_list(real_list, pred_list, seg_num, return_total=False):  # soft dice loss
    score_list = []
    for _ in range(seg_num):
        score_list.append(np.zeros(len(real_list)))
    for idx in range(len(real_list)):
        real = real_list[idx]
        pred = pred_list[idx]
        real = np.expand_dims(real, axis=0)
        pred = np.expand_dims(pred, axis=0)
        dice_score = cal_dice_score(real, pred) # b x c
        dice_score = dice_score.mean(0)
        for i, s in enumerate(dice_score):
            score_list[i][idx] = s # the i-th label's score of data idx
    scores_mean = [score.mean() for score in score_list]
    if return_total:
        return scores_mean, score_list
    else:
        return scores_mean