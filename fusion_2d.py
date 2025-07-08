import os
from glob import glob
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import *
from data import MALC, save_nii, segmap_to_onehot
from metrics import cal_dice_score_list, cal_dice_score
from keeper_2d import KnowledgeKeeperNet
import nibabel as nib

class Fusion(nn.Module):
    def __init__(self, args, out_c):
        super().__init__()
        self.args = args
        nf = self.args.nf
        self.out_c = out_c
        middle_c = out_c // 5
        self.conv1x1_1 = nn.Conv2d(nf, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_2 = nn.Conv2d(nf, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_3 = nn.Conv2d(nf, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_4 = nn.Conv2d(nf, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_5 = nn.Conv2d(nf, middle_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.aggregation = nn.Sequential(
            nn.Conv2d(middle_c*5, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.combine = nn.Sequential(
            nn.Conv2d(out_c*2, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
        )
        self.ca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(out_c, out_c // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_c // 2, out_c),
            nn.ReLU(inplace=True)
        )
        self.ca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(out_c, out_c // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_c // 2, out_c),
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
        g1 = F.interpolate(g1, size=size, mode='bilinear')
        g2 = F.interpolate(g2, size=size, mode='bilinear')
        g3 = F.interpolate(g3, size=size, mode='bilinear')
        g4 = F.interpolate(g4, size=size, mode='bilinear')
        g5 = F.interpolate(g5, size=size, mode='bilinear')
        guide = torch.cat((g1, g2, g3, g4, g5), dim=1)
        guide = self.aggregation(guide)

        comb = torch.cat((guide, enc), dim=1)
        comb = self.combine(comb)
        comb1 = comb * (comb > 0)
        comb2 = comb * (comb < 0)
        ca1 = self.ca1(comb1).unsqueeze(-1).unsqueeze(-1)
        comb1 = comb1*ca1
        ca2 = self.ca2(comb2).unsqueeze(-1).unsqueeze(-1)
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
        self.fusion2 = Fusion(args, nf)
        self.fusion3 = Fusion(args, nf)
        self.fusion4 = Fusion(args, nf)
        self.fusion5 = Fusion(args, nf)

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
        self.plane = self.args.plane
        self.base_encoder = self.args.base_encoder
        self.base_decoder = self.args.base_decoder

    def training(self, device, fold, plane=None):
        if plane is None:
            plane = self.plane

        fold_name = 'Fold_%02d' % fold
        val_idx = [fold]

        ##### Directory
        dir_log = f'./{self.type}_Fusion_Base{self.base}'
        dir_model = f'{dir_log}/model/{fold_name}/{plane}'
        os.makedirs(dir_model, exist_ok=True)

        ##### Dataset Load
        if plane == 'sagittal':
            seg_num = 16
        else:
            seg_num = 28
        train_data_path = []
        val_data_path = []
        val_idx = [1012, 1013, 1014, 1015, 1017]
        for folder_name in sorted(os.listdir(f'{self.path_dataset}/Train')):
            if int(folder_name) in val_idx:
                val_data_path.append(f'{self.path_dataset}/Train/{folder_name}')
            else:
                train_data_path.append(f'{self.path_dataset}/Train/{folder_name}')
        train_dataset = MALC(train_data_path, train=True, seg_num=seg_num, plane=plane)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = MALC(val_data_path, seg_num=seg_num, plane=plane)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        ##### Segmentation model and Fusion modules
        if self.base == 'UNet':
            from baselines.unet_2d import SegEncoder, SegDecoder, SegLoss
        loss_func = SegLoss()
        encoder = nn.DataParallel(SegEncoder(self.args)).to(device)
        encoder.load_state_dict(torch.load(self.base_encoder))
        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=self.lr, betas=(0.9, 0.999))
        decoder = nn.DataParallel(SegDecoder(self.args, seg_num)).to(device)
        decoder.load_state_dict(torch.load(self.base_decoder))
        optimizer_D = torch.optim.Adam(decoder.parameters(), lr=self.lr, betas=(0.9, 0.999))

        fusion = nn.DataParallel(FusionModules(self.args)).to(device)
        fusion.apply(weights_init_normal)
        optimizer_F = torch.optim.Adam(fusion.parameters(), lr=self.lr, betas=(0.9, 0.999))

        ##### Pretrained model (Keeper)
        path_keeper = f'./{self.type}_Keeper/model/{fold_name}/{plane}'
        keeper = nn.DataParallel(KnowledgeKeeperNet(self.args, return_diff=True)).to(device)
        keeper.load_state_dict(torch.load(f'{path_keeper}/knowledge_keeper.pth'))
        for param in keeper.parameters():
            param.requires_grad = False

        ##### Training
        best = {'epoch': 0, 'score': 0}
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
                cls_weight = Variable(batch['w']).to(device)
                loss_total = loss_func(pred_y, real_y, cls_weight)
                optimizer_E.zero_grad()
                optimizer_F.zero_grad()
                optimizer_D.zero_grad()
                loss_total.backward()
                optimizer_E.step()
                optimizer_F.step()
                optimizer_D.step()

            real_y_list, pred_y_list, index_list = self.prediction_guided(val_dataloader, seg_num, encoder, decoder, keeper, fusion, device)
            val_scores_list = cal_dice_score_list(real_y_list, pred_y_list, seg_num)
            val_score = 0
            for score in val_scores_list:
                val_score = val_score + score
            if best['score'] < val_score:
                torch.save(encoder.state_dict(), f'{dir_model}/seg_encoder.pth')
                torch.save(decoder.state_dict(), f'{dir_model}/seg_decoder.pth')
                torch.save(fusion.state_dict(), f'{dir_model}/fusion.pth')
                best['epoch'] = epoch
                best['score'] = val_score

    def testing(self, device):  # view aggregation
        dir_log = f'./{self.type}_Fusion_Base{self.base}'
        dir_all = f'{dir_log}/result_all'
        os.makedirs(dir_all, exist_ok=True)

        if self.base == 'UNet':
            from baselines.unet_2d import SegEncoder, SegDecoder

        logger_all, stream_handler_all, file_handler_all = logger_setting(file_name=f'{dir_all}/log_all.log')
        logger_all.info('Fold | Patient | Label | Score')
        fold_names = sorted(os.listdir(f'{dir_log}/model'))

        seg_num_sag = 16
        seg_num_whole = 28

        for fold_name in fold_names:
            planes = ['sagittal', 'axial', 'coronal']
            patient_prob = {}
            for i, plane in enumerate(planes):
                path_fusion = f'{dir_log}/model/{fold_name}/{plane}'
                encoder = nn.DataParallel(SegEncoder(self.args)).to(device)
                encoder_dict = torch.load(f'{path_fusion}/seg_encoder.pth')
                encoder.load_state_dict(encoder_dict)
                decoder = nn.DataParallel(SegDecoder(self.args, seg_num)).to(device)
                decoder_dict = torch.load(f'{path_fusion}/seg_decoder.pth')
                decoder.load_state_dict(decoder_dict)
                fusion = nn.DataParallel(FusionModules(self.args)).to(device)
                fusion_dict = torch.load(f'{path_fusion}/fusion.pth')
                fusion.load_state_dict(fusion_dict)
                
                path_keeper = f'./{self.type}_Keeper/model/{fold_name}/{plane}'
                keeper = nn.DataParallel(KnowledgeKeeperNet(self.args)).to(device)
                keeper_dict = torch.load(f'{path_keeper}/knowledge_keeper.pth')
                keeper.load_state_dict(keeper_dict)

                if plane == 'sagittal':
                    seg_num = seg_num_sag
                else:
                    seg_num = seg_num_whole
                test_data_path = sorted(glob(f'{self.path_dataset}/Test/*'))
                test_dataset = MALC(test_data_path, seg_num=seg_num, plane=plane)
                test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                real_y_list, pred_y_list, index_list = self.prediction_guided(test_dataloader, seg_num, encoder, decoder, keeper, fusion, device)
                blank_arr = test_dataset.blank_vox_list(vox_dim=seg_num)
                patient_ids = test_dataset.patient_list()
                whole_voxel_prob = slice_to_whole(blank_arr, pred_y_list, index_list, plane, prob_argmax=False)

                for i_, voxel in enumerate(whole_voxel_prob):
                    patient_id = patient_ids[i_]
                    if plane == 'sagittal':
                        voxel = sagittal_remap(voxel, seg_num_whole) * 0.2
                    else:
                        voxel *= 0.4
                    if patient_id in patient_prob.keys():
                        patient_prob[patient_id] += voxel
                    else:
                        patient_prob[patient_id] = voxel
            
            for patient_id, prob in patient_prob.items():
                pred = np.argmax(prob, axis=0)
                save_nii(pred.astype(float), f'{dir_all}/{fold_name}_{patient_id}_pred_y.nii.gz')
                pred = segmap_to_onehot(pred, seg_num_whole)
                real_data_path = f'{self.path_dataset}/Test/{patient_id}/{patient_id}_glm_27.nii.gz'
                real = nib.load(real_data_path).get_fdata()
                real = segmap_to_onehot(real, seg_num_whole)
                scores = cal_dice_score(np.expand_dims(real, axis=0), np.expand_dims(pred, axis=0))
                for label_id, score in enumerate(scores.squeeze(0)):
                    logger_all.info(f'{fold_name} | {patient_id} | {label_id} | {score}')
        logger_closing(logger_all, stream_handler_all, file_handler_all)

    def prediction_guided(self, dataloader, seg_num, encoder, decoder, keeper, fusion, device):
        real_y_list = []
        pred_y_list = []
        index_list = []
        encoder.eval()
        decoder.eval()
        fusion.eval()
        with torch.no_grad():
            for batch in dataloader:
                real_x = Variable(batch['x']).to(device)
                real_y = Variable(batch['y']).to(device)
                guide_list = keeper(real_x)
                enc_list = encoder(real_x)
                enc_list = fusion(guide_list, enc_list)
                pred_y = decoder(enc_list)
                pred_y = F.softmax(pred_y, dim=1)
                pred_y = torch.argmax(pred_y, dim=1, keepdim=False)
                real_y = real_y.cpu().detach().numpy()
                pred_y = pred_y.cpu().detach().numpy()
                for idx in range(pred_y.shape[0]):
                    real_y_list.append(real_y[idx])
                    pred_y_ = pred_y[idx]
                    pred_y_ = segmap_to_onehot(pred_y_.squeeze(), seg_num)
                    pred_y_list.append(pred_y_)
                    index_list.append(batch['index'][idx])
        return real_y_list, pred_y_list, index_list
