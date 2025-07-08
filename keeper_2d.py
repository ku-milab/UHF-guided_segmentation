import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
from utils import *
from data import Paired3T7T_2D, save_nii
import nibabel as nib
from metrics import cal_psnr_ssim_list
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim
from teacher_2d import TeacherEncoder, TeacherDecoder, FeatureRecon
 
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1, stack=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(inplace=True)
        ]
        if stack:
            layers.extend([
                nn.Conv2d(out_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(inplace=True)
            ])
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        out = self.model(x)
        return out

class UpMerge(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv = ConvBlock(out_c*2, out_c, kernel=kernel, pad=pad)
    def forward(self, x, skip_x):
        x = F.upsample(x, size=skip_x.shape[2:], mode='bilinear')
        x = self.conv1x1(x)
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv(x)
        return x

# https://github.com/13952522076/PRM/blob/master/models/mnasnet_prm.py
class PRMLayer(nn.Module):
    def __init__(self, mode='dotproduct'):
        super(PRMLayer, self).__init__()
        self.mode = mode
        self.max_pool = nn.AdaptiveMaxPool2d(1,return_indices=True)
        self.weight = Parameter(torch.zeros(1,1,1))
        self.bias = Parameter(torch.ones(1,1,1))
        self.sig = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.one = Parameter(torch.ones(1,1))
        self.zero = Parameter(torch.zeros(1,1))
        self.theta = Parameter(torch.rand(1,2,1,1))
        self.scale = Parameter(torch.ones(1))

    def forward(self, x):
        b, c, h, w = x.shape
        position_mask = self.get_position_mask(x, b, c, h, w) # Output (b, 2, h, w)

        # Similarity function
        query_value, query_position = self.get_query_position(x, b, c, h, w)
        query_value = query_value.view(b, -1, 1)
        x_value = x.view(b, -1, h*w)
        similarity_max = self.get_similarity(x_value, query_value, mode=self.mode)
        similarity_gap = self.get_similarity(x_value, self.gap(x).view(b, -1, 1), mode=self.mode)

        Distance = abs(position_mask - query_position)
        Distance = Distance.type(query_value.type())
        distribution = Normal(0, self.scale)
        Distance = distribution.log_prob(Distance * self.theta).exp().clone()
        Distance = (Distance.mean(dim=1)).view(b, h*w)
        similarity_max = similarity_max * Distance

        similarity = similarity_max*self.zero+similarity_gap*self.one

        context = similarity - similarity.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True) + 1e-5
        context = (context/std).view(b, h, w)
        # affine function
        context = context * self.weight + self.bias
        context = context.view(b, 1, h, w).expand(b, c, h, w).reshape(b, c, h, w)
        context = self.sig(context)
        value = x*context
        return value, context

    def get_position_mask(self, x, b, c, h, w):
        mask = (torch.ones((h, w))).nonzero().cuda()
        mask = (mask.reshape(h, w, 2)).permute(2, 0, 1).expand(b, 2, h, w)
        return mask

    def get_query_position(self, x, b, c, h, w):
        sumvalue = x.sum(dim=1, keepdim=True)
        maxvalue, maxposition = self.max_pool(sumvalue)
        t_position = torch.cat((maxposition//w, maxposition//h), dim=1)
        t_value = x[torch.arange(b),:,t_position[:,0,0,0],t_position[:,1,0,0]]
        return t_value, t_position

    def get_similarity(self, query, key_value, mode='dotproduct'):
        if mode == 'dotproduct':
            similarity = torch.matmul(key_value.permute(0, 2, 1), query).squeeze(dim=1)
        elif mode == 'l1norm':
            similarity = -(abs(query - key_value)).sum(dim=1)
        elif mode == 'gaussian':
            # Gaussian Similarity (No recommanded, too sensitive to noise)
            similarity = torch.exp(torch.matmul(key_value.permute(0, 2, 1), query))
            similarity[similarity == float("Inf")] = 0
            similarity[similarity <= 1e-9] = 1e-9
        elif mode == 'cosine':
            cos = nn.CosineSimilarity(dim=1)
            similarity = cos(query, key_value)
        else:
            similarity = torch.matmul(key_value.permute(0, 2, 1), query)
        return similarity

class GuideBlock(nn.Module):
    def __init__(self, in_c_x, in_c_skip, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv1x1_x = nn.Sequential(
            nn.Conv2d(in_c_x, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv_skip = ConvBlock(in_c_skip, out_c, kernel=kernel, pad=pad)
        self.attention = nn.Sequential(
            nn.Conv2d(out_c*2, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.prm = PRMLayer()
    def forward(self, x, skip_x):
        b, c, h, w = skip_x.shape
        x = F.upsample(x, size=(h, w), mode='bilinear')
        x = self.conv1x1_x(x)
        skip_x = self.conv_skip(skip_x)
        att = torch.cat((x, skip_x), dim=1)
        att = self.attention(att)
        x = x * att[:, 0].view(b, 1, h, w) + skip_x * att[:, 1].view(b, 1, h, w)
        out, x = self.prm(x)
        return out, x

class FeatureExtractionBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, pad):
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c, kernel=kernel, pad=pad)
        self.conv2 = ConvBlock(in_c+out_c, out_c, kernel=kernel, pad=pad)
        self.conv3 = ConvBlock(in_c+out_c*2, out_c, kernel=kernel, pad=pad)
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = torch.cat([x, c1], dim=1)
        c2 = self.conv2(c2)
        c3 = torch.cat([x, c1, c2], dim=1)
        c3 = self.conv3(c3)
        c4 = torch.cat([x, c1, c2, c3], dim=1)
        return c4

class KnowledgeKeeperNet(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf
        nf_down = self.args.nf // 4
        extract_k, extract_p = 7, 3
        transition_k, transition_s, transition_p = 4, 2, 1
        guide_k, guide_p = 7, 3
        num_conv = 3 # the number of ConvBlock in DenseBlock

        self.conv1 = FeatureExtractionBlock(in_c, nf_down, kernel=extract_k, pad=extract_p)
        self.conv2 = FeatureExtractionBlock(nf_down, nf_down, kernel=extract_k, pad=extract_p)
        self.conv3 = FeatureExtractionBlock(nf_down, nf_down, kernel=extract_k, pad=extract_p)
        self.conv4 = FeatureExtractionBlock(nf_down, nf_down, kernel=extract_k, pad=extract_p)
        self.conv5 = FeatureExtractionBlock(nf_down, nf_down, kernel=extract_k, pad=extract_p)

        self.tran1 = ConvBlock(in_c+nf_down*num_conv, nf_down, kernel=transition_k, stride=transition_s, pad=transition_p, stack=False)
        self.tran2 = ConvBlock(nf_down+nf_down*num_conv, nf_down, kernel=transition_k, stride=transition_s, pad=transition_p, stack=False)
        self.tran3 = ConvBlock(nf_down+nf_down*num_conv, nf_down, kernel=transition_k, stride=transition_s, pad=transition_p, stack=False)
        self.tran4 = ConvBlock(nf_down+nf_down*num_conv, nf_down, kernel=transition_k, stride=transition_s, pad=transition_p, stack=False)
        
        self.guide5_conv = ConvBlock(nf_down+nf_down*num_conv, nf, kernel=guide_k, pad=guide_p)
        self.guide5_PRM = PRMLayer()
        self.guide4 = GuideBlock(nf, nf_down+nf_down*num_conv, nf, kernel=guide_k, pad=guide_p)
        self.guide3 = GuideBlock(nf, nf_down+nf_down*num_conv, nf, kernel=guide_k, pad=guide_p)
        self.guide2 = GuideBlock(nf, nf_down+nf_down*num_conv, nf, kernel=guide_k, pad=guide_p)
        self.guide1 = GuideBlock(nf, in_c+nf_down*num_conv, nf, kernel=guide_k, pad=guide_p)

    def forward(self, x):
        c1 = self.conv1(x)
        t1 = self.tran1(c1)
        c2 = self.conv2(t1)
        t2 = self.tran2(c2)
        c3 = self.conv3(t2)
        t3 = self.tran3(c3)
        c4 = self.conv4(t3)
        t4 = self.tran4(c4)
        c5 = self.conv5(t4)
        c5 = self.guide5_conv(c5)
        g5, c5 = self.guide5_PRM(c5)
        g4, c4 = self.guide4(g5, c4)
        g3, c3 = self.guide3(g4, c3)
        g2, c2 = self.guide2(g3, c2)
        g1, c1 = self.guide1(g2, c1)
        enc_list = [g1, g2, g3, g4, g5]
        return enc_list

class Discriminator(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.conv1 = ConvBlock(in_c, nf, kernel=4, stride=2, pad=1, stack=False)
        self.conv2 = ConvBlock(nf, nf*2, kernel=4, stride=2, pad=1, stack=False)
        self.conv3 = ConvBlock(nf*2, nf*4, kernel=4, stride=2, pad=1, stack=False)
        self.conv4 = ConvBlock(nf*4, nf*8, kernel=4, stride=2, pad=1, stack=False)
        self.out = nn.Sequential(
            nn.Conv2d(nf*8, 1, kernel_size=4, padding=1, bias=False)
        )
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        out = self.out(c4)
        return out

class Implementation(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.type = self.args.type
        self.path_dataset = self.args.path_dataset_Paired
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.lr = self.args.lr
        self.lambda_adv = self.args.lambda_adv
        self.lambda_vox = self.args.lambda_vox
        self.plane = self.args.plane

    def training(self, device, fold, plane=None):
        if plane is None:
            plane = self.plane

        fold_name = 'Fold_%02d' % fold
        val_idx = [fold]

        ##### Directory
        dir_log = f'./{self.type}_Keeper'
        dir_model = f'{dir_log}/model/{fold_name}/{plane}'
        os.makedirs(dir_model, exist_ok=True)

        ##### Dataset Load
        train_data_path = []
        val_data_path = []
        for folder_name in sorted(os.listdir(self.path_dataset)):
            _, patient_id = folder_name.split('_')  # folder_name example: S_01
            if int(patient_id) in val_idx:
                val_data_path.append(f'{self.path_dataset}/{folder_name}')
            else:
                train_data_path.append(f'{self.path_dataset}/{folder_name}')
        train_dataset = Paired3T7T_2D(train_data_path, train=True, plane=plane)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = Paired3T7T_2D(val_data_path, plane=plane)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        ##### Initialize
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        loss_L1 = nn.L1Loss()
        loss_MSE = nn.MSELoss()

        ##### Model
        keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
        discriminator = nn.DataParallel(Discriminator(self.args)).to(device)
        keeper.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        optimizer_K = torch.optim.Adam(keeper.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.9, 0.999))

        ##### Pretrained model (Teacher)
        path_teacher = f'{self.type}_Teacher/model/{fold_name}'
        teacher_enc = nn.DataParallel(TeacherEncoder(self.args)).to(device)
        teacher_dec = nn.DataParallel(TeacherDecoder(self.args)).to(device)
        teacher_enc.load_state_dict(torch.load(f'{path_teacher}/teacher_encoder.pth'))
        teacher_dec.load_state_dict(torch.load(f'{path_teacher}/teacher_decoder.pth'))
        for param in teacher_enc.parameters():
            param.requires_grad = False
        for param in teacher_dec.parameters():
            param.requires_grad = False

        ##### Training
        best = {'epoch': 0, 'psnr': 0, 'ssim': 0}
        for epoch in tqdm(range(1, self.epochs + 1), desc='Epoch'):
            keeper.train()
            discriminator.train()

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)

                # ------------------------------
                # Discriminator
                # ------------------------------
                pred_y = teacher_dec(keeper(real_x))
                pred_real = discriminator(real_y)
                valid = Variable(Tensor(np.ones(pred_real.size())), requires_grad=False)
                loss_D_real = loss_MSE(pred_real, valid)
                pred_syn = discriminator(pred_y.detach())
                syn = Variable(Tensor(np.zeros(pred_syn.size())), requires_grad=False)
                loss_D_syn = loss_MSE(pred_syn, syn)
                loss_D = self.lambda_adv * (loss_D_real + loss_D_syn)
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                # ------------------------------
                # Keeper
                # ------------------------------
                # Distillation loss
                teach_1, teach_2, teach_3, teach_4, teach_5 = teacher_enc(real_y) # feature maps of teacher encoder
                guided_1, guided_2, guided_3, guided_4, guided_5 = keeper(real_x) # feature maps of knowledge keeper
                loss_K_distill_L2 = loss_MSE(guided_1, teach_1) + loss_MSE(guided_2, teach_2) + loss_MSE(guided_3, teach_3) + loss_MSE(guided_4, teach_4) + loss_MSE(guided_5, teach_5)
                loss_K_distill_match = loss_dist_match(guided_1, teach_1) + loss_dist_match(guided_2, teach_2) + loss_dist_match(guided_3,teach_3) + loss_dist_match(guided_4, teach_4) + loss_dist_match(guided_5, teach_5)

                # Voxel-wise loss
                fake_y = teacher_dec([guided_1, guided_2, guided_3, guided_4, guided_5])
                loss_K_vox = self.lambda_vox * loss_L1(fake_y, real_y)

                # Adversarial loss
                pred_fake = discriminator(fake_y)
                loss_K_adv = self.lambda_adv * loss_MSE(pred_fake, valid)

                # Total loss
                loss_K = loss_K_distill_L2 + loss_K_distill_match + loss_K_vox + loss_K_adv

                optimizer_K.zero_grad()
                loss_K.backward()
                optimizer_K.step()

            real_y_list, pred_y_list, index_list = self.prediction(val_dataloader, keeper, teacher_dec, device)
            val_psnr, val_ssim = cal_psnr_ssim_list(real_y_list, pred_y_list)
            if best['psnr'] < val_psnr and best['ssim'] < val_ssim:
                torch.save(keeper.state_dict(), f'{dir_model}/knowledge_keeper.pth')
                torch.save(discriminator.state_dict(), f'{dir_model}/discriminator.pth')
                best['epoch'] = epoch
                best['psnr'] = val_psnr
                best['ssim'] = val_ssim

    def testing(self, device):  # view aggregation
        dir_log = f'./{self.type}_Keeper'
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)

        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        logger_all.info('[Fold | Patient ID | PSNR | SSIM]')
        fold_names = sorted(os.listdir(f'{dir_log}/model'))

        for fold_name in fold_names:
            planes = ['sagittal', 'axial', 'coronal']
            patient_value = {}
            for i, plane in enumerate(planes):
                path_teacher = f'./{self.type}_Teacher/model/{fold_name}/{plane}'
                teacher_dec_dict = torch.load(f'{path_teacher}/teacher_decoder.pth')
                teacher_dec = nn.DataParallel(TeacherDecoder(self.args)).to(device)
                teacher_dec.load_state_dict(teacher_dec_dict)
                path_keeper = f'{dir_log}/model/{fold_name}/{plane}'
                keeper_dict = torch.load(f'{path_keeper}/knowledge_keeper.pth')
                keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
                keeper.load_state_dict(keeper_dict)
                test_data_path = [f'{self.path_dataset}/S_{fold_name[-2:]}']
                test_dataset = Paired3T7T_2D(test_data_path, plane=plane)
                test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                real_y_list, pred_y_list, index_list = self.prediction(test_dataloader, keeper, teacher_dec, device)
                blank_arr = test_dataset.blank_vox_list()
                patient_ids = test_dataset.patient_list()
                whole_voxel_value = slice_to_whole(blank_arr, pred_y_list, index_list, plane)
                for i_, voxel in enumerate(whole_voxel_value):
                    patient_id = patient_ids[i_]
                    if patient_id in patient_value.keys():
                        patient_value[patient_id] += voxel
                    else:
                        patient_value[patient_id] = voxel
            for patient_id, value in patient_value.items():
                value /= 3
                value = value.squeeze()
                save_nii(value, f'{dir_all}/{patient_id}_pred_y.nii.gz')
                real_data_path = f'{self.path_dataset}/{patient_id}/7t_02_norm.nii'
                real = nib.load(real_data_path).get_fdata()
                psnr = cal_psnr(real, value)
                ssim = cal_ssim(real, value)
                logger_all.info(f'{fold_name} | {patient_id} | {psnr} | {ssim}')
        logger_closing(logger_all, stream_handler_all, file_handler_all)

    def prediction(self, dataloader, keeper, teacher_decoder, device):
        real_y_list = []
        pred_y_list = []
        index_list = []
        keeper.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)
                pred_y = teacher_decoder(keeper(real_x))
                real_y = real_y.cpu().detach().numpy()
                pred_y = pred_y.cpu().detach().numpy()
                for idx in range(pred_y.shape[0]):
                    real_y_ = real_y[idx].squeeze()
                    pred_y_ = pred_y[idx].squeeze()
                    real_y_list.append(real_y_)
                    pred_y_list.append(pred_y_)
                    index_list.append(batch['index'][idx])
        return real_y_list, pred_y_list, index_list
