import os
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *
from data import Paired3T7T_3D, save_nii
from metrics import cal_psnr_ssim_list
 
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        out = self.model(x)
        return out

class UpMerge(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv = ConvBlock(out_c*2, out_c, kernel=kernel, pad=pad)
    def forward(self, x, skip_x):
        x = F.upsample(x, size=skip_x.shape[2:], mode='trilinear')
        x = self.conv1x1(x)
        x = torch.cat((x, skip_x), dim=1)
        x = self.conv(x)
        return x

class TeacherEncoder(nn.Module):
    def __init__(self, args, in_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.pooling = nn.MaxPool3d(kernel_size=2)
        self.conv1 = ConvBlock(in_c, nf)
        self.conv2 = ConvBlock(nf, nf*2)
        self.conv3 = ConvBlock(nf*2, nf*4)
        self.conv4 = ConvBlock(nf*4, nf*8)
        self.conv5 = ConvBlock(nf*8, nf*16)
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)
        c2 = self.conv2(p1)
        p2 = self.pooling(c2)
        c3 = self.conv3(p2)
        p3 = self.pooling(c3)
        c4 = self.conv4(p3)
        p4 = self.pooling(c4)
        c5 = self.conv5(p4)
        enc_list = [c1, c2, c3, c4, c5]
        return enc_list

class ReconBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv = ConvBlock(out_c, out_c, kernel=kernel, pad=pad)
    def forward(self, x, to_size):
        x = F.upsample(x, size=to_size, mode='trilinear')
        x = self.conv1x1(x)
        x = self.conv(x)
        return x

class FeatureRecon(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.recon1 = ReconBlock(nf*2, nf)
        self.recon2 = ReconBlock(nf*4, nf*2)
        self.recon3 = ReconBlock(nf*8, nf*4)
        self.recon4 = ReconBlock(nf*16, nf*8)
    def forward(self, enc_list):
        c1, c2, c3, c4, c5 = enc_list
        a1 = self.recon1(c2, c1.shape[2:])
        a2 = self.recon2(c3, c2.shape[2:])
        a3 = self.recon3(c4, c3.shape[2:])
        a4 = self.recon4(c5, c4.shape[2:])
        recon_list = [a1, a2, a3, a4]
        return recon_list

class TeacherDecoder(nn.Module):
    def __init__(self, args, out_c=1):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.conv = ConvBlock(nf*16, nf*16)
        self.up1 = UpMerge(nf*16, nf*8)
        self.up2 = UpMerge(nf*8, nf*4)
        self.up3 = UpMerge(nf*4, nf*2)
        self.up4 = UpMerge(nf*2, nf)
        self.out = nn.Sequential(
            nn.Conv3d(nf, out_c, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, enc_list):
        c1, c2, c3, c4, c5 = enc_list
        c5 = self.conv(c5)
        u1 = self.up1(c5, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = self.out(u4)
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
        self.lambda_img = self.args.lambda_img

    def training(self, device, fold):
        fold_name = 'Fold_%02d' % fold
        val_idx = [fold]

        ##### Directory
        dir_log = f'./{self.type}_Teacher'
        dir_model = f'{dir_log}/model/{fold_name}'
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
        train_dataset = Paired3T7T_3D(train_data_path, train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = Paired3T7T_3D(val_data_path)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        ##### Initialize
        loss_L1 = nn.L1Loss()
        loss_MSE = nn.MSELoss()

        ##### Model
        encoder = nn.DataParallel(TeacherEncoder(self.args))
        decoder = nn.DataParallel(TeacherDecoder(self.args))
        featrecon = nn.DataParallel(FeatureRecon(self.args))
        encoder.apply(weights_init_normal)
        decoder.apply(weights_init_normal)
        featrecon.apply(weights_init_normal)
        encoder.to(device)
        decoder.to(device)
        featrecon.to(device)
        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimizer_D = torch.optim.Adam(decoder.parameters(), lr=self.lr, betas=(0.9, 0.999))
        optimizer_F = torch.optim.Adam(featrecon.parameters(), lr=self.lr, betas=(0.9, 0.999))

        ##### Training
        best = {'epoch': 0, 'psnr': 0, 'ssim': 0}
        for epoch in tqdm(range(1, self.epochs + 1), desc='Epoch'):
            encoder.train()
            decoder.train()
            featrecon.train()

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                real_y = Variable(batch['y']).to(device)
                enc_list = encoder(real_y)
                frecon_list = featrecon(enc_list)
                loss_FR = loss_MSE(enc_list[0], frecon_list[0]) + loss_MSE(enc_list[1], frecon_list[1]) + loss_MSE(enc_list[2], frecon_list[2]) + loss_MSE(enc_list[3], frecon_list[3])
                #### Add noise
                for list_idx in range(len(enc_list)):
                    noised = enc_list[list_idx] + torch.rand(enc_list[list_idx].shape).to(device) * 1. + 0.
                    enc_list[list_idx] = noised
                pred_y = decoder(enc_list)
                loss_T = self.lambda_img * loss_L1(real_y, pred_y) + loss_FR
                optimizer_E.zero_grad()
                optimizer_D.zero_grad()
                optimizer_F.zero_grad()
                loss_T.backward()
                optimizer_E.step()
                optimizer_D.step()
                optimizer_F.step()

            real_y_list, pred_y_list, _ = self.prediction(val_dataloader, encoder, decoder, device)
            val_psnr, val_ssim = cal_psnr_ssim_list(real_y_list, pred_y_list)
            if best['psnr'] < val_psnr and best['ssim'] < val_ssim:
                torch.save(encoder.state_dict(), f'{dir_model}/teacher_encoder.pth')
                torch.save(decoder.state_dict(), f'{dir_model}/teacher_decoder.pth')
                torch.save(featrecon.state_dict(), f'{dir_model}/feature_recon.pth')
                best['epoch'] = epoch
                best['psnr'] = val_psnr
                best['ssim'] = val_ssim

    def testing(self, device):
        dir_log = f'./{self.type}_Teacher'
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)

        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        logger_all.info('[Fold | Patient ID | PSNR | SSIM]')
        fold_names = sorted(os.listdir(f'{dir_log}/model'))

        for fold_name in fold_names:
            dir_model = f'{dir_log}/model/{fold_name}'
            encoder_dict = torch.load(f'{dir_model}/teacher_encoder.pth')
            decoder_dict = torch.load(f'{dir_model}/teacher_decoder.pth')
            encoder = nn.DataParallel(TeacherEncoder(self.args)).to(device)
            decoder = nn.DataParallel(TeacherDecoder(self.args)).to(device)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            data_path = [f'{self.path_dataset}/S_{fold_name[-2:]}']
            dataset = Paired3T7T_3D(data_path)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            real_y, pred_y, patient_ids = self.prediction(dataloader, encoder, decoder, device, dir_all)
            mean_psnr, mean_ssim, total_psnr, total_ssim = cal_psnr_ssim_list(real_y, pred_y, return_total=True)
            for idx, patient_id in enumerate(patient_ids):
                logger_all.info(f'{fold_name} | {patient_id} | {total_psnr[idx]} | {total_ssim[idx]}')
                
        logger_closing(logger_all, stream_handler_all, file_handler_all)

    def prediction(self, dataloader, encoder, decoder, device, save_pred_path=False):
        patient_ids = []
        real_y_list = []
        pred_y_list = []
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_y = Variable(batch['y']).to(device)
                pred_y = decoder(encoder(real_y))
                real_y = real_y.cpu().detach().numpy()
                pred_y = pred_y.cpu().detach().numpy()
                for idx in range(pred_y.shape[0]):
                    patient_id = str(batch['patient_id'][idx])
                    patient_ids.append(patient_id)
                    real_y_ = real_y[idx].squeeze()
                    pred_y_ = pred_y[idx].squeeze()
                    real_y_list.append(real_y_)
                    pred_y_list.append(pred_y_)
                    if save_pred_path:
                        save_nii(pred_y_, f'{save_pred_path}{patient_id}_recon_y.nii.gz')
        return real_y_list, pred_y_list, patient_ids

