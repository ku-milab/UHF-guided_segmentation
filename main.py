import os
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='3D') # 3D / 2D
parser.add_argument('--mode', type=str, default='all') # all / train / test
parser.add_argument('--net', type=str, default='T') # T(teacher) / K(keeper) / F(fusion for segmentation)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--path_dataset_Paired', type=str, default='/PATH_PAIRED')
parser.add_argument('--path_dataset_IBSR', type=str, default='/PATH_IBSR')
parser.add_argument('--path_dataset_MALC', type=str, default='/PATH_MALC')
parser.add_argument('--base', type=str, default='UNet') # segmentation models (./baselines)
parser.add_argument('--base_encoder', type=str, default='BASELINE_ENCODER.pth')
parser.add_argument('--base_decoder', type=str, default='BASELINE_DECODER.pth')
parser.add_argument('--plane', type=str, default='axial') # plane for a 2D-model: axial / coronal / sagittal
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--nf', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lambda_img', type=int, default=100)
parser.add_argument('--lambda_vox', type=int, default=100)
parser.add_argument('--lambda_adv', type=int, default=0.5)
args = parser.parse_args()

if __name__ == "__main__":
    devices = "%d" % args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = list(range(1, 16))

    if args.mode == 'all':
        if args.type == '2D':
            from teacher_2d import Implementation as Teacher
            from keeper_2d import Implementation as Keeper
            from fusion_2d import Implementation as Fusion
            teacher = Teacher(args)
            keeper = Keeper(args)
            fusion = Fusion(args)
            planes = ['sagittal', 'axial', 'coronal']
            for fold in folds:
                for plane in planes:
                    teacher.training(device, fold, plane)
                teacher.testing(device)
                for plane in planes:
                    keeper.training(device, fold, plane)
                keeper.testing(device)
                for plane in planes:
                    fusion.training(device, fold, plane)
                fusion.testing(device)
        elif args.type == '3D':
            from teacher_3d import Implementation as Teacher
            from keeper_3d import Implementation as Keeper
            from fusion_3d import Implementation as Fusion
            teacher = Teacher(args)
            keeper = Keeper(args)
            fusion = Fusion(args)
            for fold in folds:
                teacher.training(device, fold)
                teacher.testing(device)
                keeper.training(device, fold)
                keeper.testing(device)
                fusion.training(device, fold)
                fusion.testing(device)

    else:
        if args.type == '2D':
            if args.net == 'T':
                from teacher_2d import Implementation as Model
            elif args.net == 'K':
                from keeper_2d import Implementation as Model
            elif args.net == 'F':
                from fusion_2d import Implementation as Model
        elif args.type == '3D':
            if args.net == 'T':
                from teacher_3d import Implementation as Model
            elif args.net == 'K':
                from keeper_3d import Implementation as Model
            elif args.net == 'F':
                from fusion_3d import Implementation as Model
        model = Model(args)
        if args.mode == 'train':
            for fold in folds:
                model.training(device, fold)
        elif args.mode == 'test':
            model.testing(device)
