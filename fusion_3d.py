import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *
from data import IBSR, save_nii, segmap_to_onehot
from metrics import cal_dice_score_list
from keeper_3d import KnowledgeKeeperNet

class Fusion(nn.Module):
    def __init__(self, args, out_c):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.out_c = out_c
        middle_c = out_c // 5
        self.conv1x1_1 = nn.Conv3d(nf, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_2 = nn.Conv3d(nf*2, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_3 = nn.Conv3d(nf*4, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_4 = nn.Conv3d(nf*8, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_5 = nn.Conv3d(nf*16, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.aggregation = nn.Sequential(
            nn.Conv3d(middle_c*5, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.combine = nn.Sequential(
            nn.Conv3d(out_c*2, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_c)
        )
        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten(),
            nn.Linear(out_c, out_c//2),
            nn.ReLU(inplace=True),
            nn.Linear(out_c//2, out_c),
            nn.ReLU(inplace=True)
        )
        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten(),
            nn.Linear(out_c, out_c//2),
            nn.ReLU(inplace=True),
            nn.Linear(out_c//2, out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, guide_feat, enc):
        g1, g2, g3, g4, g5 = guide_feat
        size = enc.shape[2:]
        g1 = self.conv1x1_1(g1)
        g2 = self.conv1x1_2(g2)
        g3 = self.conv1x1_3(g3)
        g4 = self.conv1x1_4(g4)
        g5 = self.conv1x1_5(g5)
        g1 = F.interpolate(g1, size=size, mode='trilinear')
        g2 = F.interpolate(g2, size=size, mode='trilinear')
        g3 = F.interpolate(g3, size=size, mode='trilinear')
        g4 = F.interpolate(g4, size=size, mode='trilinear')
        g5 = F.interpolate(g5, size=size, mode='trilinear')
        guide = torch.cat((g1, g2, g3, g4, g5), dim=1)
        guide = self.aggregation(guide)

        comb = torch.cat((guide, enc), dim=1)
        comb = self.combine(comb)
        comb1 = comb * (comb > 0)
        comb2 = comb * (comb < 0)
        ca1 = self.ca1(comb1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        comb1 = comb1*ca1
        ca2 = self.ca2(comb2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        comb2 = comb2*ca2
        comb_attn = comb1 + comb2

        out = enc + comb_attn
        return comb_attn, out


class FusionModules(nn.Module):
    def __init__(self, args, return_gating=False):
        super().__init__()
        self.return_gating = return_gating
        self.args = args
        nf = self.args.nf
        self.fusion1 = Fusion(args, nf)
        self.fusion2 = Fusion(args, nf*2)
        self.fusion3 = Fusion(args, nf*4)
        self.fusion4 = Fusion(args, nf*8)
        self.fusion5 = Fusion(args, nf*16)

    def forward(self, guide_feat, enc_feat):
        e1, e2, e3, e4, e5 = enc_feat
        g1, f1 = self.fusion1(guide_feat, e1)
        g2, f2 = self.fusion2(guide_feat, e2)
        g3, f3 = self.fusion3(guide_feat, e3)
        g4, f4 = self.fusion4(guide_feat, e4)
        g5, f5 = self.fusion5(guide_feat, e5)
        out = [f1, f2, f3, f4, f5]
        if self.return_gating:
            gating = [g1, g2, g3, g4, g5]
            return gating, out
        else:
            return out


class Implementation(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.path_dataset = self.args.path_dataset_MALC
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.lr = self.args.lr
        self.base = self.args.base
        self.base_encoder = self.args.base_encoder
        self.base_decoder = self.args.base_decoder
        self.seg_num = 4

    def training(self, device, fold):
        fold_name = 'Fold_%02d' % fold
        val_idx = [fold]

        ##### Directory
        dir_log = f'./{self.type}_Fusion_Base{self.base}'
        dir_model = f'{dir_log}/model/{fold_name}'
        os.makedirs(dir_model, exist_ok=True)

        ##### Dataset Load
        val_idx = [1, 3, 18]
        train_idx = [2, 4, 5, 6, 7, 8, 9]
        test_idx = [10, 11, 12, 13, 14, 15, 16, 17]
        train_data_path = []
        val_data_path = []
        test_data_path = []
        for folder_name in sorted(os.listdir(self.path_dataset_IBSR)):
            _, patient_id = folder_name.split('_')  # folder_name example: IBSR_01
            if int(patient_id) in val_idx:
                val_data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')
            elif int(patient_id) in train_idx:
                train_data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')
            elif int(patient_id) in test_idx:
                test_data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')
        train_dataset = IBSR(train_data_path, train=True, seg_type=self.seg_type, seg_num=self.seg_num)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = IBSR(val_data_path, seg_type=self.seg_type, seg_num=self.seg_num)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        ##### Segmentation model and Fusion modules
        if self.base == 'UNet':
            from baselines.unet_3d import SegEncoder, SegDecoder
            from torch.nn import CrossEntropyLoss as SegLoss
        loss_func = SegLoss()
        encoder = nn.DataParallel(SegEncoder(self.args)).to(device)
        encoder.load_state_dict(torch.load(self.base_encoder))
        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=self.lr_fusion, betas=(0.9, 0.999))        
        decoder = nn.DataParallel(SegDecoder(self.args, self.seg_num)).to(device)
        decoder.load_state_dict(torch.load(self.base_decoder))
        optimizer_D = torch.optim.Adam(decoder.parameters(), lr=self.lr_fusion, betas=(0.9, 0.999))

        fusion = nn.DataParallel(FusionModules(self.args)).to(device)
        fusion.apply(weights_init_normal)
        optimizer_F = torch.optim.Adam(fusion.parameters(), lr=self.lr_fusion, betas=(0.9, 0.999))
        
        ##### Pretrained model (Keeper)
        path_keeper = f'./{self.type}_Keeper/model/{fold_name}'
        keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
        keeper.load_state_dict(torch.load(f'{path_keeper}/knowledge_keeper.pth'))
        for param in keeper.parameters():
            param.requires_grad = False

        ##### Training
        best = {'epoch': 0, 'score': 0, 'loss': np.inf}
        for epoch in tqdm(range(1, self.epochs + 1), desc='Epoch'):
            encoder.train()
            decoder.train()
            fusion.train()

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)
                guide_list = keeper(real_x)
                enc_list = encoder(real_x)
                enc_list = fusion(guide_list, enc_list)
                pred_y = decoder(enc_list)
                loss_total = loss_func(pred_y, real_y)
                optimizer_E.zero_grad()
                optimizer_F.zero_grad()
                optimizer_D.zero_grad()
                loss_total.backward()
                optimizer_E.step()
                optimizer_F.step()
                optimizer_D.step()

            real_y_list, pred_y_list, _ = self.prediction_guided(val_dataloader, encoder, decoder, keeper, fusion, device)
            val_scores_list = cal_dice_score_list(real_y_list, pred_y_list, self.seg_num)
            val_score = 0
            for score in val_scores_list:
                val_score = val_score + score
            if best['score'] < val_score:
                torch.save(encoder.state_dict(), f'{dir_model}/seg_encoder.pth')
                torch.save(decoder.state_dict(), f'{dir_model}/seg_decoder.pth')
                torch.save(fusion.state_dict(), f'{dir_model}/fusion.pth')
                best['epoch'] = epoch
                best['score'] = val_score

    def testing(self, device):
        dir_log = f'./{self.type}_Fusion_Base{self.base}'
        dir_all = f'{dir_log}/result_all/'
        os.makedirs(dir_all, exist_ok=True)

        if self.base == 'UNet':
            from baselines.unet_3d import SegEncoder, SegDecoder

        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all_.log')
        logger_all.info('[Fold | Patient ID | Background | CSF | GM | WM]')
        fold_names = sorted(os.listdir(f'{dir_log}/model'))

        for fold_name in fold_names:
            path_fusion = f'{dir_log}/model/{fold_name}'
            encoder_dict = torch.load(f'{path_fusion}/seg_encoder.pth')
            encoder = nn.DataParallel(SegEncoder(self.args)).to(device)
            encoder.load_state_dict(encoder_dict)
            decoder_dict = torch.load(f'{path_fusion}/seg_decoder.pth')
            decoder = nn.DataParallel(SegDecoder(self.args, self.seg_num)).to(device)
            decoder.load_state_dict(decoder_dict)
            fusion_dict = torch.load(f'{path_fusion}/fusion.pth')
            fusion = nn.DataParallel(FusionModules(self.args)).to(device)
            fusion.load_state_dict(fusion_dict) 

            path_keeper = f'./{self.type}_Keeper/model/{fold_name}'
            keeper_dict = torch.load(f'{path_keeper}/knowledge_keeper.pth')
            keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
            keeper.load_state_dict(keeper_dict)

            test_idx = [10, 11, 12, 13, 14, 15, 16, 17]
            test_data_path = []
            for folder_name in sorted(os.listdir(self.path_dataset_IBSR)):
                _, patient_id = folder_name.split('_')  # folder_name example: IBSR_01
                if int(patient_id) in test_idx:
                    test_data_path.append(f'{self.path_dataset_IBSR}/{folder_name}')
            test_dataset = IBSR(test_data_path, seg_type=self.seg_type, seg_num=self.seg_num)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            real_y_list, pred_y_list, patient_ids = self.prediction_guided(test_dataloader, encoder, decoder, keeper, fusion, device, save_pred_path=f'{dir_all}/{fold_name}_')
            scores_mean, score_list = cal_dice_score_list(real_y_list, pred_y_list, self.seg_num, return_total=True)
            scores_bg, scores_CSF, scores_GM, scores_WM = score_list

            for idx, patient_id in enumerate(patient_ids):
                score_bg = scores_bg[idx]
                score_CSF = scores_CSF[idx]
                score_GM = scores_GM[idx]
                score_WM = scores_WM[idx]
                logger_all.info(f'{fold_name} | {patient_id} | {score_bg} | {score_CSF} | {score_GM} | {score_WM}')
        logger_closing(logger_all, stream_handler_all, file_handler_all)

    def prediction_guided(self, dataloader, encoder, decoder, keeper, fusion, device, save_pred_path=False):
        patient_ids = []
        real_y_list = []
        pred_y_list = []
        encoder.eval()
        decoder.eval()
        fusion.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)
                g1, g2, g3, g4, g5 = keeper(real_x)
                e1, e2, e3, e4, e5 = encoder(real_x)
                enc_list = fusion([g1, g2, g3, g4, g5], [e1, e2, e3, e4, e5])
                pred_y = decoder(enc_list)
                pred_y = F.softmax(pred_y, dim=1)
                pred_y = torch.argmax(pred_y, dim=1, keepdim=False)
                real_y = real_y.cpu().detach().numpy()
                pred_y = pred_y.cpu().detach().numpy()
                for idx in range(pred_y.shape[0]):
                    patient_id = str(batch['patient_id'][idx])
                    patient_ids.append(patient_id)
                    real_y_list.append(real_y[idx])
                    pred_y_ = pred_y[idx]
                    pred_y_ = segmap_to_onehot(pred_y_.squeeze(), self.seg_num)
                    pred_y_list.append(pred_y_)
                    if save_pred_path:
                        save_nii(np.argmax(pred_y_, axis=0).astype(float), f'{save_pred_path}{patient_id}_pred_y.nii.gz')
        return real_y_list, pred_y_list, patient_ids
