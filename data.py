import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage import rotate
from skimage import exposure

def save_nii(arr, path, affine=np.eye(4)):
    nii_img = nib.Nifti1Image(arr, affine=affine)
    nib.save(nii_img, path)

def load_nii(path_file):
    proxy = nib.load(path_file)
    array = proxy.get_fdata()
    return array

class Paired3T7T_3D(Dataset):
    def __init__(self, path_dataset, train=False):
        self.patient_ids, self.x_list, self.y_list = self.get_dataset(path_dataset)
        self.train = train

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        x = self.x_list[index]
        y = self.y_list[index]
        x_h, x_w, x_d = x.shape
        y_h, y_w, y_d = y.shape
        if self.train:
            aug_index = np.random.rand(3)
            data = [x, y]
            if aug_index[0] > 0.5:
                x, y = rand_rotate_specific(data)
            if aug_index[1] > 0.5:
                x, y = rand_scale(data, 0.8, 1.2)
            if aug_index[2] > 0.5:
                x, y = flip_by_axis(data, axis=0)
        x = torch.from_numpy(x.copy()).float().view(1, x_h, x_w, x_d)
        if self.seg:
            y = segmap_to_onehot(y) # output shape: (num_labels, h, w, d)
            y = torch.from_numpy(y.copy()).float()
        else:
            y = torch.from_numpy(y.copy()).float().view(1, y_h, y_w, y_d)
        return {'patient_id': patient_id, 'x': x, 'y': y}

    def get_dataset(self, path_dataset):
        patient_ids = []
        x_list = []
        y_list = []
        for path_data in path_dataset:
            patient_id = path_data.split('/')[-1]
            path_3t = f'{path_data}/3t_norm.nii'
            path_7t = f'{path_data}/7t_norm.nii'
            x = nib.load(path_3t).get_data()
            y = nib.load(path_7t).get_data()
            x_list.append(x)
            y_list.append(y)
            patient_ids.append(patient_id)
        return patient_ids, x_list, y_list

class Paired3T7T_2D(Dataset):
    def __init__(self, path_dataset, train=False, plane='axial'):
        self.patient_ids, self.idx_list, self.x_list, self.y_list = self.get_dataset(path_dataset, plane)
        self.path_dataset = path_dataset
        self.train = train
        self.plane = plane

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, index):
        data_index = self.idx_list[index]
        x = self.x_list[index]
        y = self.y_list[index]
        h, w = x.shape
        if self.train:
            aug_index = np.random.rand(3)
            data = [x, y]
            if aug_index[0] > 0.5:
                x, y = rand_rotate_specific(data)
            if aug_index[1] > 0.5:
                x, y = rand_scale(data, 0.8, 1.2, type='2D')
            if aug_index[2] > 0.5:
                x, y = flip_by_axis(data, axis=0)
        x = torch.from_numpy(x.copy()).float().view(1, h, w)
        if self.seg:
            y = segmap_to_onehot(y)
            y = torch.from_numpy(y.copy()).float()
        else:
            y = torch.from_numpy(y.copy()).float().view(1, h, w)
        return {'index': data_index, 'x': x, 'y': y}

    def blank_vox_list(self, vox_dim=1):
        vox_list = []
        for path_data in self.path_dataset:
            patient_id = path_data.split('/')[-1]
            path_x = f'{path_data}/3t_norm.nii'
            x = nib.load(path_x).get_data()
            vox_list.append(np.zeros((vox_dim, *x.shape)))
        return vox_list

    def patient_list(self):
        return self.patient_ids

    def get_dataset(self, path_dataset, plane):
        patient_ids = []
        idx_list = []
        x_list = []
        y_list = []
        for patient_idx, path_data in enumerate(path_dataset):
            patient_id = path_data.split('/')[-1]
            patient_ids.append(patient_id)
            path_3t = f'{path_data}/3t_norm.nii'
            path_7t = f'{path_data}/7t_norm.nii'
            x = nib.load(path_3t).get_data()
            y = nib.load(path_7t).get_data()
            if plane == 'axial':
                x = np.transpose(x, (2, 0, 1))
                y = np.transpose(y, (2, 0, 1))
            elif plane == 'coronal':
                x = np.transpose(x, (1, 0, 2))
                y = np.transpose(y, (1, 0, 2))

            for slice_idx in range(x.shape[0]):
                idx_list.append(np.array([patient_idx, slice_idx]))
                x_list.append(x[slice_idx])
                y_list.append(y[slice_idx])
        return patient_ids, idx_list, x_list, y_list

def segmap_to_onehot(y, num_labels=4):
    out = np.zeros((num_labels, *y.shape))
    for idx, label in enumerate(range(num_labels)):
        out[idx] = np.where(y == label, 1, 0)
    return out

def flip_by_axis(data, axis):
    out = []
    for d in data:
        out.append(np.flip(d, axis=axis))
    return out

def rand_scale(data, range_min, range_max, type='3D'):
    scale_uniform = np.random.rand(1)[0]
    scale = (range_max - range_min) * scale_uniform + range_min
    if type == '3D':
        aff_matrix = np.array([[scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, scale]])
    elif type == '2D':
        aff_matrix = np.array([[scale, 0],
                            [0, scale]])
    center = 0.5 * np.array(data[0].shape)
    offset = center - center.dot(aff_matrix)
    out = []
    for d in data:
        out.append(affine_transform(d, aff_matrix, offset=offset))
    return out

def rand_rotate_specific(data, specific_angle=[90, 180, 270]):
    angle = np.random.choice(specific_angle)
    out = []
    for d in data:
        out.append(rotate(d, angle, reshape=False))
    return out

def clahe(arr, clip_limit=0.2, type='3D'):
    if type == '3D':
        kernel_size = (arr.shape[0] // 5,
            arr.shape[1] // 5,
            arr.shape[2] // 5)
    elif type == '2D':
        kernel_size = (arr.shape[0] // 5,
                arr.shape[1] // 5)
    kernel_size = np.array(kernel_size)
    out = [exposure.equalize_adapthist(im,
                                 kernel_size=kernel_size,
                                 clip_limit=clip_limit)
             for im in [arr]]
    return out[0]

# https://www.nitrc.org/projects/ibsr
class IBSR(Dataset):
    def __init__(self, path_dataset, train=False, seg_num=4):
        self.patient_ids, self.x_list, self.y_list = self.get_dataset_IBSR(path_dataset)
        self.seg_num = seg_num
        self.train = train

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        x = self.x_list[index]
        y = self.y_list[index]
        x = clahe(x)
        if self.train:
            aug_index = np.random.rand(3)
            data = [x, y]
            if aug_index[0] > 0.5:
                x, y = rand_rotate_specific(data)
            if aug_index[1] > 0.5:
                x, y = rand_scale(data, 0.8, 1.2)
            if aug_index[2] > 0.5:
                x, y = flip_by_axis(data, axis=0)
        x_h, x_w, x_d = x.shape
        x = torch.from_numpy(x.copy()).float().view(1, x_h, x_w, x_d)
        y = segmap_to_onehot(y, self.seg_num) # output shape: (num_labels, h, w, d)
        y = torch.from_numpy(y.copy()).float()
        if self.train == False:
            ulb = self.ulb_list[index]
            ulb = torch.from_numpy(ulb.copy()).float().view(1, x_h, x_w, x_d)
            return {'patient_id': patient_id, 'x': x, 'y': y, 'ulb': ulb}
        else:
            return {'patient_id': patient_id, 'x': x, 'y': y}

    def get_dataset_IBSR(self, path_dataset):
        patient_ids = []
        x_list = []
        y_list = []
        for path_data in path_dataset:
            patient_id = path_data.split('/')[-1].split('_')[-1]
            path_x = f'{path_data}/IBSR_{patient_id}_ana_strip_norm.nii'
            # The 'fill' files have any regions of zeros that are inside the brain mask set to 1 (the CSF value). https://www.nitrc.org/forum/message.php?msg_id=25702
            path_y = f'{path_data}/IBSR_{patient_id}_segTRI_fill_ana.nii'
            x = nib.load(path_x).get_fdata().squeeze()
            y = nib.load(path_y).get_fdata().squeeze()
            x_list.append(x)
            y_list.append(y)
            patient_ids.append(patient_id)
        return patient_ids, x_list, y_list

class MALC(Dataset):
    def __init__(self, path_dataset, train=False, seg_num=28, plane='axial', spatial_info=False):
        self.patient_ids, self.idx_list, self.x_list, self.y_list, self.w_list = self.get_dataset_MALC(path_dataset, plane, spatial_info)
        self.path_dataset = path_dataset
        self.seg_num = seg_num
        self.train = train
        self.plane = plane
        self.spatial_info = spatial_info

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, index):
        data_idx = self.idx_list[index]
        x = self.x_list[index]
        y = self.y_list[index]
        w = self.w_list[index]
        if self.spatial_info:
            x = torch.from_numpy(x.copy()).float()
        else:
            x_h, x_w = x.shape
            x = torch.from_numpy(x.copy()).float().view(1, x_h, x_w)
        y = segmap_to_onehot(y, self.seg_num) # output shape: (num_labels, h, w)
        y = torch.from_numpy(y.copy()).float()
        w = torch.from_numpy(w.copy()).float()
        return {'index': data_idx, 'x': x, 'y': y, 'w': w}

    def blank_vox_list(self, vox_dim=1):
        vox_list = []
        for path_data in self.path_dataset:
            patient_id = path_data.split('/')[-1]
            path_x = f'{path_data}/{patient_id}_norm.nii.gz'
            x = nib.load(path_x).get_data()
            vox_list.append(np.zeros((vox_dim, *x.shape)))
        return vox_list

    def patient_list(self):
        return self.patient_ids

    def get_dataset_MALC(self, path_dataset, plane, spatial_info):
        patient_ids = []
        idx_list = []
        x_list = []
        y_list = []
        w_list = []
        for patient_idx, path_data in enumerate(path_dataset):
            patient_id = path_data.split('/')[-1]
            patient_ids.append(patient_id)
            path_x = f'{path_data}/{patient_id}_norm.nii.gz'
            path_y = f'{path_data}/{patient_id}_glm_27.nii.gz'
            x = nib.load(path_x).get_fdata().squeeze()
            y = nib.load(path_y).get_fdata().squeeze()
            if plane == 'sagittal':
                lut_aseg = np.zeros(int(y.max() + 1), dtype='int')
                labels_sag = np.array([0, 3, 4, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
                for idx, value in enumerate(labels_sag):
                    lut_aseg[value] = idx # lut_aseg => [0, 0, 0, 3, 4, 0, ..., 27]
                left_right = {1: 3, 2: 4, 5: 18, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 15: 25, 16: 26, 17: 27}
                for key, value in left_right.items():
                    y[y == key] = value
                mapped_aseg_sag = lut_aseg.ravel()[y.astype(int).ravel()]
                y = mapped_aseg_sag.reshape((y.shape[0], y.shape[1], y.shape[2]))
            w = self.create_weight_mask(y.astype(int))  # weight mask
            if plane == 'axial':
                x = np.transpose(x, (2, 0, 1))
                y = np.transpose(y, (2, 0, 1))
                w = np.transpose(w, (2, 0, 1))
            elif plane == 'coronal':
                x = np.transpose(x, (1, 0, 2))
                y = np.transpose(y, (1, 0, 2))
                w = np.transpose(w, (1, 0, 2))
            for slice_idx in range(x.shape[0]):
                idx_list.append(np.array([patient_idx, slice_idx]))
                if spatial_info:
                    if slice_idx < 3:
                        pad_num = 3 - slice_idx
                        x_ = x[0:slice_idx+4]
                        x_ = np.pad(x_, ((pad_num,0),(0,0),(0,0)), mode='constant', constant_values=0)
                    elif slice_idx > (x.shape[0]-4):
                        pad_num = 3 - (x.shape[0]-1-slice_idx)
                        x_ = x[slice_idx-3:]
                        x_ = np.pad(x_, ((0,pad_num),(0,0),(0,0)), mode='constant', constant_values=0)
                    else:
                        x_ = x[slice_idx-3:slice_idx+4]
                    x_list.append(x_)
                else:
                    x_list.append(x[slice_idx])
                y_list.append(y[slice_idx])
                w_list.append(w[slice_idx])
        return patient_ids, idx_list, x_list, y_list, w_list

    # weight map generator https://github.com/Deep-MI/FastSurfer/blob/stable/FastSurferCNN/data_loader/load_neuroimaging_data.py
    def create_weight_mask(self, mapped_aseg, max_weight=5, max_edge_weight=5):
        """
        Function to create weighted mask - with median frequency balancing and edge-weighting
        :param np.ndarray mapped_aseg: label space segmentation
        :param int max_weight: an upper bound on weight values
        :param int max_edge_weight: edge-weighting factor
        :return: np.ndarray weights_mask: generated weights mask
        """
        unique, counts = np.unique(mapped_aseg, return_counts=True)
        for i in range(np.max(unique)):
            if i not in unique:
                counts = np.insert(counts, i, 0)
        # Median Frequency Balancing
        class_wise_weights = np.median(counts) / counts
        class_wise_weights[class_wise_weights > max_weight] = max_weight
        (h, w, d) = mapped_aseg.shape
        weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))
        # Gradient Weighting
        (gx, gy, gz) = np.gradient(mapped_aseg)
        grad_weight = max_edge_weight * np.asarray(
            np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
            dtype='float')
        weights_mask += grad_weight
        return weights_mask
