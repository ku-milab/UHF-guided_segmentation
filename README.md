# UHF-guided Segmentation

This is the PyTorch implementation of our preprint paper "Transferring Ultra-high Field Representations for Intensity-Guided Brain Segmentation of Low Field MRI".

---

### Architecture

TBU

---

### Requirements

torch==1.10.1

torchvision==0.11.2

scikit-image==0.19.1

scikit-learn==1.0.2

nibabel==3.2.1

nilearn==0.8.1

scipy==1.7.3

---

### Usage

Command format:
```
python main.py --type <3D / 2D> --mode <all / train / test> --net <T / K / F> --gpu <GPU_NUMBER> \\
--path_dataset_Paired <path to the paired 3T and 7T dataset> --path_dataset_IBSR <path to the IBSR dataset for tissue segmentation> \\
--path_dataset_MALC <path to the MALC dataset for region segmentation> --base <name of the baseline segmentation model> \\
--base_encoder <path to a pre-trained weight file of the segmentation encoder> --base_decoder <path to a pre-trained weight file of the segmentation decoder> \\
--plane <plane for a 2D model: axial / coronal / sagittal>
```

1. For training teacher and knowledge keeper networks for a 3D version, you can use the following command:
```
python main.py --type 3D --mode train --net T --gpu 1 --path_dataset_Paired /PATH_PAIRED && python main.py --type 3D --net K --mode train --gpu 1 --path_dataset_Paired /PATH_PAIRED
```

2. For training fusion modules by using 3D U-Net as the baseline segmentation model, you can use the following command:
```
python main.py --type 3D --mode train --net F --gpu 1 --path_dataset_IBSR /PATH_IBSR --base UNet --base_encoder /PATH_BASE/UNET_ENCODER.pth --base_decoder /PATH_BASE/UNET_DECODER.pth
```

3. To implement the above two steps at once, you can use the following command:
```
python main.py --type 3D --mode all --gpu 1 --path_dataset_Paired /PATH_PAIRED --path_dataset_IBSR /PATH_IBSR --base UNet --base_encoder /PATH_BASE/UNET_ENCODER.pth --base_decoder /PATH_BASE/UNET_DECODER.pth
```
   
