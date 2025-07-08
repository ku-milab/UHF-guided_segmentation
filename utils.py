import logging
import torch
from torch import nn
import numpy as np
 
def logger_setting(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=file_name)
    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger, stream_handler, file_handler

def logger_closing(logger, stream_handler, file_handler):
    stream_handler.close()
    logger.removeHandler(stream_handler)
    file_handler.close()
    logger.removeHandler(file_handler)
    del logger, stream_handler, file_handler

def weights_init_normal(m, activation='leaky_relu'):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity=activation)
    elif classname.find('ConvTranspose3d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity=activation)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity=activation)
    elif classname.find('BatchNorm3d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity=activation)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity=activation)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def match_size_3D(x, size):
    _, _, h1, w1, d1 = x.shape
    h2, w2, d2 = size
    while d1 != d2:
        if d1 < d2:
            x = nn.functional.pad(x, (0, 1), mode='constant', value=0)
            d1 += 1
        else:
            x = x[:, :, :, :, :d2]
            break
    while w1 != w2:
        if w1 < w2:
            x = nn.functional.pad(x, (0, 0, 0, 1), mode='constant', value=0)
            w1 += 1
        else:
            x = x[:, :, :, :w2, :]
            break
    while h1 != h2:
        if h1 < h2:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
            h1 += 1
        else:
            x = x[:, :, :h2, :, :]
            break
    return x

def match_size_2D(x, size):
    _, _, h1, w1 = x.shape
    h2, w2 = size
    while w1 != w2:
        if w1 < w2:
            x = nn.functional.pad(x, (0, 1), mode='constant', value=0)
            w1 += 1
        else:
            x = x[:, :, :, :w2]
            break
    while h1 != h2:
        if h1 < h2:
            x = nn.functional.pad(x, (0, 0, 0, 1), mode='constant', value=0)
            h1 += 1
        else:
            x = x[:, :, :h2, :]
            break
    return x

def loss_dist_match(real, fake):
    loss = 0
    loss_MSE = nn.MSELoss()
    for b in range(real.shape[0]):
        real_vol = real[b]
        fake_vol = fake[b]
        real_std, real_mu = torch.std_mean(real_vol, dim=0, unbiased=False)
        fake_std, fake_mu = torch.std_mean(fake_vol, dim=0, unbiased=False)
        loss += loss_MSE(real_std, fake_std) + loss_MSE(real_mu, fake_mu)
    return loss

def slice_to_whole(blank_arr, slice_list, index_list, plane, prob_argmax=False):
    for i, index in enumerate(index_list): # index[0]: patient index, index[1]: slice index
        slice = slice_list[i]
        if prob_argmax:
            slice = np.argmax(slice, axis=0)
        if plane == 'axial':
            blank_arr[index[0]][:, :, :, index[1]] = slice
        elif plane == 'coronal':
            blank_arr[index[0]][:, :, index[1], :] = slice
        elif plane == 'sagittal':
            blank_arr[index[0]][:, index[1], :, :] = slice

def sagittal_remap(pred_prob, origin_seg_num):
    c, h, w, d = pred_prob.shape
    out = np.zeros((origin_seg_num, h, w, d))
    labels_sag = [0, 3, 4, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    left_right = {1: 3, 2: 4, 5: 18, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 15: 25, 16: 26, 17: 27}
    right_left = dict((value, key) for (key, value) in left_right.items())
    for idx in range(c):
        origin_label = labels_sag[idx]
        if origin_label in left_right.values():
            origin_label_left = right_left[origin_label]
            out[origin_label_left] = pred_prob[idx]
            out[origin_label] = pred_prob[idx]
        else:
            out[origin_label] = pred_prob[idx]
    return out
