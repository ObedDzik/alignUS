import os

import warnings

warnings.filterwarnings("ignore")

import pprint
from glob import glob
import multiprocessing
from itertools import chain

import matplotlib
import pylab as plt
from PIL import Image
from scipy.signal import hilbert

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from transformers import SamProcessor

# torch.multiprocessing.set_sharing_strategy('file_system')

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
# from PIL import Image
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise, Pool

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


DATASET_DIR = '/projects/bk_pca'
DROPPED_ROWS = [
    # some needles are not presented
    'pat27_cor9', 'pat71_cor0', 'pat111_cor3', 'pat7_cor6',
    # no intersection with the prostate mask
    'pat9_cor0', 'pat39_core0', 'pat39_cor2',
    # strange needle length
    'pat128_cor6',
    # involvement mismatched
    'pat2200_cor3', 'pat168_cor7',
]
# DROPPED_ROWS = []
__all__ = ['DATASETS']


class BkDataset(Dataset):
    def __init__(self,
                 metadata_path=r'/h/minht/projects/teusformer/data_splits/bk_Paul_splits/metadata.csv',
                 transform=None,
                 fold_idx=0,
                 set_name='train',
                 patch_dim=(48, 32, 0),  # patch-size, vertical stride, horizontal stride
                 in_prostate=True,
                 patch_gen_config=None,
                 in_channel=1,
                 input_size=(32, 32),
                 min_inv=0.0,
                 min_gs=0,
                 patch_save_dir=None,
                 save_patches=False,
                 oversampling=False,
                 *args, **kwargs,
                 ):
        super().__init__()
        self.transform = transform
        self.patch_dim = patch_dim
        self.in_prostate = in_prostate
        self.fold_idx, self.set_name, self.metadata_path = fold_idx, set_name, metadata_path
        self.metadata = self.load_metadata(min_inv, min_gs)
        self.patch_save_dir = patch_save_dir
        self.save_patches = save_patches
        self.patch_gen_config = patch_gen_config
        self.in_channel = in_channel
        self.input_size = input_size
        self.to_tensor = None
        self.oversampling = oversampling
        self.warning = ParamWarnings()

    def _transform(self, x, roi=None, wp=None):
        if self.transform is None:
            if roi is not None and wp is not None:
                return x, roi, wp
            return x

        if roi is not None:
            assert wp is not None
            if self.in_prostate:
                roi = roi * wp
            seg_maps = SegmentationMapsOnImage(np.array([roi, wp]).transpose([1, 2, 0]),
                                               shape=x.shape)
            x, seg_maps_aug = self.transform(image=x, segmentation_maps=seg_maps)
            roi, wp = seg_maps_aug.get_arr().transpose([2, 0, 1])
            return x, roi, wp
        else:
            return self.transform(image=x)

    def load_metadata(self, min_inv=0.0, min_gs=0):
        df = (pd.read_csv(self.metadata_path)
              .groupby('fold_idx').get_group(self.fold_idx)
              .groupby('set_name').get_group(self.set_name)
              .reset_index())
        df.drop(['index'], axis=1, inplace=True)
        df.drop(df[df.filetemplate.isin(DROPPED_ROWS)].index, axis=0, inplace=True)
        if min_inv > 0:
            df = df.loc[(df.Involvement >= min_inv) | (df.Involvement == 0)]
        if min_gs > 0:
            df = df.loc[(df.GleasonScore >= min_gs) | (df.GleasonScore == 0)]
        df['c_retain'] = False

        return df

    def _load_numpy(self, idx, verbose=False, skip_rf=False, keyword: str = None):
        info = self.metadata.iloc[idx]
        files = glob(f'{DATASET_DIR}/BK_{info["center"]}_CORES/{info["filetemplate"]}_*')
        try:
            assert len(files) == 5
        except:
            print(files)
            return
        fd = {}  # file dict
        for file in files:
            k = os.path.basename(file.split('.')[0].split('_')[-1])
            fd[k] = file
        if verbose:
            print(fd)
        if keyword is not None:
            if keyword == 'rf':
                return np.load(fd[keyword], mmap_mode='c')  # [..., 100:200]
            return np.load(fd[keyword], mmap_mode='r')

        if info.center == 'UBC':
            roi = np.load(f'/projects/bk_pca/patches/corrected_roi/UBC/{info["filetemplate"]}_needle.npy',
                          mmap_mode='r')
        else:
            roi = np.load(fd['needle'], mmap_mode='r')
        # roi = np.load(fd['needle'], mmap_mode='r')
        wp = np.load(fd['prostate'], mmap_mode='r')
        if skip_rf:
            return roi, wp
        rf = np.load(fd['rf'], mmap_mode='c').astype('float32')  # [..., 100:200].astype('float32')
        return rf, roi, wp

    def load_numpy(self, idx, verbose=False):
        """idx could be both integer (index) or string (core ID)"""
        idx = self._id_idx(idx)
        return self._load_numpy(idx, verbose)

    def id2idx(self, _id):
        return self.metadata[self.metadata.id == _id].index[0]

    def _id_idx(self, idx):
        if isinstance(idx, str):
            idx = self.id2idx(idx)
        return idx

    def show_raw_rf(self, idx, target_shape=None, fig_size=(5, 5), verbose=False, fig_idx=0,
                    patch_masks=None, patch_locs=None, patch_idx=None, convert_whole_rf=False,
                    patch_dim=None):
        rf, roi, wp = self._load_numpy(idx)
        if verbose:
            print('RF shape: ', rf.shape)
        if patch_locs is not None:
            try:
                assert patch_masks is None
            except AssertionError:
                raise ValueError('Does not accept both patch_masks and patch_locs simultaneously.')
            patch_masks = self.locs2masks(patch_locs, roi=roi, patch_dim=patch_dim)

        plot_raw_rf(rf, contours=[roi, wp], frm_idx=-1, target_shape=target_shape, fig_size=fig_size, i=fig_idx,
                    patch_masks=patch_masks, patch_idx=patch_idx, convert_whole_rf=convert_whole_rf,
                    )

    def core_info_from_id(self, _id):
        return self.metadata.loc[self.metadata['id'] == _id]

    def _gen_patch_loc_per_core(self, idx, patch_dim=None, in_prostate=None):
        roi, wp = self._load_numpy(idx, skip_rf=True)
        in_prostate = self.in_prostate if in_prostate is None else in_prostate
        if in_prostate:
            roi = roi * wp
        locs = np.argwhere(roi == 1)  #
        rows, cols = locs[:, 0], locs[:, 1]
        patch_size, stride_v, stride_h = self.patch_dim if patch_dim is None else patch_dim
        rows = np.arange(min(rows), max(rows), stride_v)
        if stride_h > 0:
            leftmost, rightmost = [], []
            for r in rows:
                _lm = locs[locs[:, 0] == r, 1]
                if len(_lm):
                    leftmost.append(min(_lm))
                    rightmost.append(max(_lm))
            leftmost, rightmost = np.array(leftmost), np.array(rightmost)
            # leftmost = np.array([min(locs[locs[:, 0] == r, 1]) for r in rows])
            # rightmost = np.array([max(locs[locs[:, 0] == r, 1]) for r in rows])
            max_dist = max(rightmost - leftmost)
            shifts = range(0, max_dist, stride_h)
            cols, shifts = (np.repeat(leftmost, len(shifts)),
                            np.repeat(shifts, len(leftmost)).reshape([-1, len(leftmost)]).T.flatten())
            cols = cols + shifts
            rows = np.repeat(rows, len(cols) // len(leftmost))
        else:
            cols = np.array([locs[locs[:, 0] == r, 1].mean() for r in rows])
            cols = np.round(cols[np.invert(np.isnan(cols))]).astype('int')

        patch_locs = []
        for i, (r, c) in enumerate(zip(rows, cols)):
            r, c = max(r, patch_size // 2), max(c, patch_size // 2)
            patch_locs.append((r, c))

        return patch_locs

    @staticmethod
    def _set_patch_locs(func):
        def _setter(self, patch_locs, *args, **kwargs):
            patch_locs = [patch_locs] if isinstance(patch_locs, tuple) and len(patch_locs) == 2 else patch_locs
            return func(self, patch_locs, *args, **kwargs)

        return _setter

    @_set_patch_locs
    def locs2masks(self, patch_locs, mask_size=None, roi=None, patch_dim=None):
        assert mask_size is not None or roi is not None
        patch_masks = []
        patch_size, stride_v, stride_h = self.patch_dim if patch_dim is None else patch_dim
        for i, (r, c) in enumerate(patch_locs):
            patch_mask = np.zeros_like(roi) if mask_size is None else np.zeros(mask_size)
            r, c = max(r, patch_size // 2), max(c, patch_size // 2)
            patch_mask[r - patch_size // 2: r + patch_size // 2, c - patch_size // 2: c + patch_size // 2] = 1
            patch_masks.append(patch_mask)
        return patch_masks

    @_set_patch_locs
    def locs2patches(self, patch_locs, image, patch_dim=None):
        patch_size, stride_v, stride_h = self.patch_dim if patch_dim is None else patch_dim
        patches = []
        for i, (r, c) in enumerate(patch_locs):
            r, c = max(r, patch_size // 2), max(c, patch_size // 2)
            patches.append(image[r - patch_size // 2: r + patch_size // 2, c - patch_size // 2: c + patch_size // 2])
        if len(patches) > 1:
            return np.array(patches).squeeze()
        return patches[-1]

    def loc2patch(self, patch_loc, image, patch_dim=None):
        patch_size, stride_v, stride_h = self.patch_dim if patch_dim is None else patch_dim
        r, c = patch_loc
        r, c = max(r, patch_size // 2), max(c, patch_size // 2)
        return image[r - patch_size // 2: r + patch_size // 2, c - patch_size // 2: c + patch_size // 2]

    def gen_patch_loc_per_core(self, idx, patch_dim=None, in_prostate=None):
        """idx could be both integer (index) or string (core ID)"""
        idx = self._id_idx(idx)
        return self._gen_patch_loc_per_core(idx, patch_dim, in_prostate)

    def oversampling_minor_class(self):
        assert len(self.metadata.TrueLabel.unique()) == 2
        n_benign = (self.metadata.TrueLabel == 0).sum()
        n_cancer = (self.metadata.TrueLabel == 1).sum()
        duplicate_ratio = round(max(n_benign, n_cancer) / min(n_benign, n_cancer))
        self.metadata = pd.concat(
            [self.metadata] +
            [self.metadata.loc[self.metadata.TrueLabel == (n_cancer == min(n_benign, n_cancer))]]
            * duplicate_ratio)

    def __getitem__(self, idx):
        rf, roi, wp = self._load_numpy(idx)
        info = self.metadata.iloc[idx]
        rf, roi, wp = self._transform(rf), self._transform(roi), self._transform(wp)
        if self.to_tensor is not None:
            rf = self.to_tensor(rf)
        return rf, int(info.TrueLabel)

    def __len__(self):
        return len(self.metadata)

    def describe(self, verbose=False, set_name=''):
        metadata = self.metadata
        core_specifiers, idx = np.unique(metadata.id, return_index=True)
        label = np.array(metadata.TrueLabel)
        n_cores = len(core_specifiers)
        n_benign_core, n_cancer_core = sum(label[idx] == 0), sum(label[idx] == 1)
        patient_id = [str(pid) + c for pid, c in zip(metadata.PatientId, metadata.center)]
        pid, pid_idx = np.unique(patient_id, return_index=True)
        n_patients = len(pid)
        n_benign_patient, n_cancer_patient = sum(label[pid_idx] == 0), sum(label[pid_idx] == 1)
        stats = {
            'set_name': set_name,
            'n_patches': len(label),
            'n_patches_benign': sum(label == 0),
            'n_patches_cancer': sum(label == 1),
            'n_cores': n_cores,
            'n_benign_cores': n_benign_core,
            'n_cancer_cores': n_cancer_core,
            'n_patients': n_patients,
            'n_benign_patients': n_benign_patient,
            'n_cancer_patients': n_cancer_patient,
        }
        if verbose:
            pp = pprint.PrettyPrinter(indent=7)
            pp.pprint(stats)
        stats['core_specifiers'] = core_specifiers


def to_uint8(_x):
    _x = (_x - _x.min()) / (_x.max() - _x.min())
    _x *= 255
    return _x.astype(np.uint8)


class BkDatasetRfAatP(BkDataset):
    def __init__(self, *args, pre_crop_size=(256, 256),
                 force_resolution=False,
                 rf_save_dir='/projects/bk_pca/patches/',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_crop_size = tuple([max(s1, s2) for s1, s2 in zip(pre_crop_size, self.input_size)])
        self.rf_save_dir = rf_save_dir
        self.force_resolution = force_resolution

        self.rf_a, self.wp_cr, self.roi_cr = None, None, None
        self.prepare_rf_a()
        self.to_tensor = transforms.ToTensor()
        if self.transform is None:
            if self.set_name == 'train':
                self.transform = iaa.Sequential([
                    iaa.Fliplr(0.5),  # horizontal flips,
                    iaa.Affine(),
                    # iaa.Affine(rotate=(-20, 20)),
                    # iaa.Affine(
                    #     scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    #     rotate=(-15, 15)
                    # ),
                    # iaa.Rot90([0, 1, 3]),
                    # iaa.GaussianBlur(sigma=(0.0, 1.0)),
                    # iaa.Sharpen((0.0, 0.01)),
                    # iaa.CropToFixedSize(*self.input_size),
                    iaa.CenterCropToFixedSize(*self.input_size),
                ], random_order=False)
            else:
                self.transform = iaa.CenterCropToFixedSize(*self.input_size)
        if self.oversampling:
            self.oversampling_minor_class()

    @staticmethod
    def center_crop(x, mask):
        loc = np.argwhere(mask == 1).T
        i, j = (loc[0].min(), loc[0].max()), (loc[1].min(), loc[1].max())
        return x[i[0]:i[1], j[0]:j[1]]

    def prepare_rf_a(self):
        split_by = self.metadata_path.split('/')[-2].split('_')[-2]
        assert split_by.lower() in ['paul', 'mo']
        save_dir = f'{self.rf_save_dir}/{split_by}/fold{self.fold_idx}/{self.set_name}'
        os.makedirs(save_dir, exist_ok=True)

        for i, file in enumerate(['rf_a', 'wp', 'roi']):
            if not os.path.exists(f'{save_dir}/{file}.npy'):
                break
            if i >= 2:
                self.rf_a = np.tile(np.load(f'{save_dir}/rf_a.npy', mmap_mode='r'),
                                    [1, 1, 1, self.in_channel])
                self.wp_cr = np.load(f'{save_dir}/wp.npy', mmap_mode='r')
                self.roi_cr = np.load(f'{save_dir}/roi.npy', mmap_mode='r')
                return None

        self.rf_a = np.zeros((len(self),) + self.pre_crop_size + (1,), dtype='float32')  # self.in_channel
        self.roi_cr = np.zeros((len(self),) + self.pre_crop_size, dtype='uint8')
        self.wp_cr = np.zeros((len(self),) + self.pre_crop_size, dtype='uint8')
        rf_a_gen = RfAGenerator(self, pre_crop_size=self.pre_crop_size)

        patch_gen_dl = DataLoader(rf_a_gen, shuffle=False, drop_last=False,
                                  batch_size=self.patch_gen_config['batch_size'],
                                  num_workers=self.patch_gen_config['workers'],
                                  )
        idx = 0
        for batch in (pbar := tqdm(patch_gen_dl, mininterval=10)):
            pbar.set_description(f'Generating {self.set_name} Analytical RF...', refresh=False)
            self.rf_a[idx:idx + batch[0].shape[0]] = batch[0]
            self.roi_cr[idx:idx + batch[0].shape[0]] = batch[1]
            self.wp_cr[idx:idx + batch[0].shape[0]] = batch[2]
            idx += batch[0].shape[0]

        np.save(f'{save_dir}/rf_a.npy', self.rf_a)
        np.save(f'{save_dir}/wp.npy', self.wp_cr)
        np.save(f'{save_dir}/roi.npy', self.roi_cr)
        self.rf_a = np.tile(self.rf_a, [1, 1, 1, self.in_channel])

    def __getitem__(self, idx):
        info = self.metadata.iloc[idx]
        rf_a, roi_cr, wp_cr = self._transform(self.rf_a[idx], self.roi_cr[idx], self.wp_cr[idx])

        input_size = self.input_size[0] if self.force_resolution else 1024

        # from transformers import AutoImageProcessor
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base',
        #                                          size={"shortest_edge": input_size},
        #                                          pad_size={'height': input_size, 'width': input_size})
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

        processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base",
                                                 size={"longest_edge": input_size},
                                                 pad_size={'height': input_size, 'width': input_size})

        inputs = processor(to_uint8(rf_a), input_boxes=None, return_tensors="pt")

        inputs = {k: v.squeeze() for k, v in inputs.items()}

        # ground_truth = wp_cr
        # inputs['ground_truth'] = ground_truth
        # inputs['ground_truth_mask'] = np.ones_like(ground_truth)

        ground_truth_mask = roi_cr * wp_cr
        # ground_truth = ground_truth_mask * float(info.Involvement)
        ground_truth = ground_truth_mask * int(info.TrueLabel)
        inputs['ground_truth'] = ground_truth
        inputs['ground_truth_mask'] = ground_truth_mask
        return inputs, int(info.TrueLabel)


class BkDatasetRfAatPforSSL(BkDatasetRfAatP):
    def __getitem__(self, idx):
        x = Image.fromarray(to_uint8(self.rf_a[idx]))
        info = self.metadata.iloc[idx]
        return self.transform(x), int(info.TrueLabel)


class PatchGenerator(Dataset):
    def __init__(self, _object: BkDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object = _object

    def __getitem__(self, idx):
        obj = self._object
        try:
            info = obj.metadata.iloc[[idx]]
            # patch_locs = obj.gen_patch_loc_per_core(idx)
            # info = info.loc[info.index.repeat(len(patch_locs))]
            patch_locs = obj.gen_patch_loc_per_core(idx)
            info = info.loc[info.index.repeat(len(patch_locs))]

            info.insert(1, 'PatchId', info.id.str[:] + '_' + [str(i) for i in range(len(info))])

            if obj.patch_save_dir is not None and obj.save_patches:
                rf = obj.load_numpy(idx)[0]
                _id = obj.metadata.iloc[idx].id
                os.makedirs(obj.patch_save_dir, exist_ok=True)
                rf_p_arrays = obj.locs2patches(patch_locs, rf)
                for i, patch_id in enumerate(info['PatchId']):
                    # np.save(f'{obj.patch_save_dir}/{patch_id}.npy',
                    #         rf_p_arrays[i].astype('float32').mean(axis=-1))
                    np.save(f'{obj.patch_save_dir}/{patch_id}.npy',
                            rf_p_arrays[i].astype('int16'))  # .astype('float32')
        except Exception as e:
            print(e, info.id, 'here')
            return info, []

        return info, patch_locs

    def __len__(self):
        return len(self._object.metadata)


class RfAGenerator(Dataset):
    def __init__(self, _object: BkDatasetRfAatP, pre_crop_size=(256, 256),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._object = _object
        self.pre_crop_size = pre_crop_size

    def __getitem__(self, idx):
        obj = self._object

        rf, roi, wp = obj.load_numpy(idx)
        rf_a = cv2.resize(make_analytical(obj.center_crop(rf, wp)[..., -1:]),  # obj.in_channel
                          self.pre_crop_size,
                          interpolation=cv2.INTER_LINEAR)
        roi_cr = cv2.resize(obj.center_crop(roi.astype('uint8'), wp),
                            self.pre_crop_size,
                            interpolation=cv2.INTER_NEAREST)
        wp_cr = cv2.resize(obj.center_crop(wp.astype('uint8'), wp),
                           self.pre_crop_size,
                           interpolation=cv2.INTER_NEAREST)
        rf_a = rf_a[..., np.newaxis] if rf_a.ndim == 2 else rf_a

        return rf_a, roi_cr, wp_cr

    def __len__(self):
        return len(self._object.metadata)


class BkDatasetPatch(BkDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce_op = None
        if self.oversampling:
            self.oversampling_minor_class()

    def _get_patch(self, idx, loc=None):
        if self.patch_save_dir is not None:
            if 'PatchId' in self.metadata.columns:
                patch_id = self.metadata.iloc[idx].PatchId
                rf_p = np.load(os.path.join(self.patch_save_dir, patch_id + '.npy'), mmap_mode='r')
                return rf_p
        rf, roi, wp = self._load_numpy(idx)  # rf: HWF
        if loc is None:
            locs = np.argwhere((roi * wp) == 1)
            loc = tuple(locs[np.random.choice(len(locs))])
        return self.loc2patch(loc, rf)

    def reduce_dim(self, x):
        match self.reduce_op:
            case '1d':
                x = x.mean((0, 1))[np.newaxis]
            case '1dm':
                x = x.reshape([-1, x.shape[-1]])
            case '2d':
                x = x.mean(-1)[..., np.newaxis]
            case _:
                pass
        return x

    def __getitem__(self, idx):
        rf_p = self._get_patch(idx).astype('float32')
        info = self.metadata.iloc[idx]
        rf_p = self._transform(self.reduce_dim(rf_p))
        rf_p = norm_robust(rf_p)
        if self.to_tensor is not None:
            rf_p = self.to_tensor(rf_p)
        return rf_p, int(info.TrueLabel)

    def show_patch(self, idx, *args, **kwargs):
        info = self.metadata.loc[idx]
        core_info = self.metadata.loc[self.metadata['id'] == info['id']]
        _core_info = core_info.PatchId.reset_index()
        patch_idx = _core_info[_core_info.PatchId == info.PatchId].index.item()
        patch_locs = self.patch_locs_from_info(info)

        kwargs['patch_idx'] = patch_idx
        kwargs['patch_locs'] = patch_locs
        self.show_raw_rf(idx, *args, **kwargs)

        core_info.insert(2, 'selected', (core_info.PatchId == info.PatchId).tolist())
        return info, core_info

    def patch_id2idx(self, patch_id):
        return self.metadata[self.metadata['PatchId'] == patch_id].index[0]

    def core_info_from_patch_id(self, patch_id):
        info = self.metadata.loc[self.metadata['PatchId'] == patch_id]
        return self.core_info_from_id(info['id'].item())

    def patch_locs_from_info(self, info):
        patch_locs = [self.metadata.iloc[i]['patch_locs'] for i in
                      self.metadata[self.metadata['id'] == info['id']].index.tolist()]
        return patch_locs

    def prepare_patches(self):
        patch_locs, metadata = [], []
        patch_gen = PatchGenerator(self)

        def collate_fn(batch):
            __info, __patch_locs = [_[0] for _ in batch], [_[1] for _ in batch]
            return __info, __patch_locs

        patch_gen_dl = DataLoader(patch_gen, shuffle=False, drop_last=False,
                                  batch_size=self.patch_gen_config['batch_size'],
                                  num_workers=self.patch_gen_config['workers'],
                                  # num_workers=0,
                                  collate_fn=collate_fn,
                                  )
        for _info, _patch_locs in (pbar := tqdm(patch_gen_dl, mininterval=10)):
            pbar.set_description(f'Generating {self.set_name} patch masks...', refresh=False)
            patch_locs.append(_patch_locs)
            metadata.append(_info)

        self.metadata = pd.concat([pd.concat(_) for _ in metadata]).reset_index()
        self.metadata['patch_locs'] = list(chain(*chain(*patch_locs)))


class BkDatasetPatchTo1D(BkDatasetPatch):
    def __init__(self, *args, seq_len=100, aug_1d=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce_op = '1d'
        self.transform = self.get_patch_transform()
        self.seq_len = seq_len
        self.aug_1d = aug_1d

    def get_patch_transform(self):
        if self.set_name == 'train':
            transform = iaa.Sequential([
                # iaa.CropToFixedSize(*self.input_size),
                # iaa.Crop(percent=(0, 0.4)),
                iaa.CenterCropToFixedSize(*self.input_size),
                # iaa.Dropout2d(p=(0, 0.2)),
            ], random_order=True)
        else:
            transform = iaa.CenterCropToFixedSize(*self.input_size)
        return transform

    @staticmethod
    def get_ts_transform(seed, seq_len):
        # transform_1d = Reverse() @ 0.5
        def add_seed(func, *args, **kwargs):
            return func(*args, seed=seed, **kwargs)

        # transform_1d = add_seed(TimeWarp) @ 0.2  # add_seed(Reverse) @ 0.5
        transform_1d = (
                add_seed(Reverse) @ 0.2
                + add_seed(Crop, size=seq_len)  # random crop subsequences with length seq_len
                + add_seed(TimeWarp) @ 0.2  # random time warping 5 times in parallel
            #     + AddNoise(
            # scale=(0.01, 0.05)) @ 0.1  # with 50% probability, add random noise up to 1% - 5%# signal up to 10% - 50%
            #     + Drift(max_drift=(0.1, 0.5)) @ 0.1  # with 80% probability, random drift the
            #     + Pool(size=(1, 3)) @ 0.1
            # Quantize(n_levels=[30, ])  # random quantize to 10-, 20-, or 30- level sets
        )
        return transform_1d
        # return None

    def __getitem__(self, idx):
        rf_p = self._get_patch(idx).astype('float32')
        info = self.metadata.iloc[idx]
        rf_p = self.reduce_dim(self._transform(rf_p))
        seed = np.random.choice(int(1e6))
        if self.set_name == 'train' and self.aug_1d:
            transform_1d = BkDatasetPatchTo1D.get_ts_transform(seed, self.seq_len)
            rf_p = transform_1d.augment(rf_p).astype('float32')
        else:
            rf_p = rf_p[:, :self.seq_len].astype('float32')
        # rf_p = norm_robust(rf_p)
        # rf_p = norm_0mean(rf_p)
        return rf_p.T, int(info.TrueLabel)


class BkDatasetPatchTo1DM(BkDatasetPatchTo1D):
    """"""

    def __init__(self, *args, patch_downsampling_factor=0.25, **kwargs):
        self.pdf = patch_downsampling_factor
        super().__init__(*args, **kwargs)
        self.reduce_op = '1dm'

    def get_patch_transform(self):
        # if self.set_name == 'train':
        # transform = iaa.Sequential([
        #         iaa.CropToFixedSize(*self.input_size),
        #         iaa.Resize(self.pdf, interpolation='linear'),
        #     ], random_order=False)
        # else:
        #     transform = iaa.Sequential([
        #         iaa.CenterCropToFixedSize(*self.input_size),
        #         iaa.Resize(self.pdf, interpolation='linear'),
        #     ], random_order=False)
        transform = iaa.Sequential([
            iaa.CenterCropToFixedSize(*self.input_size),
            iaa.Resize(self.pdf, interpolation='cubic'),
        ], random_order=False)
        return transform


class BkDatasetPatchTo2D(BkDatasetPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce_op = '2d'
        self.to_tensor = transforms.ToTensor()
        if self.transform is None:
            # self.transform = iaa.CenterCropToFixedSize(*self.input_size)
            if self.set_name == 'train':
                ran_seq = iaa.Sequential([
                    iaa.Fliplr(0.5),  # horizontal flips,
                    # iaa.Affine(rotate=(-20, 20)),
                    iaa.Rot90([0, 1, 3]),
                    # iaa.GaussianBlur(sigma=(0.0, 1.0)),
                    # iaa.Sharpen((0.0, 0.01)),
                ], random_order=True)
                self.transform = iaa.Sequential([ran_seq, iaa.CenterCropToFixedSize(*self.input_size)])
            else:
                self.transform = iaa.CenterCropToFixedSize(*self.input_size)

    def __getitem__(self, idx):
        rf_p = self._get_patch(idx).astype('float32')
        info = self.metadata.iloc[idx]
        rf_p = self._transform(rf_p)
        rf_p = norm_robust(rf_p)
        if self.to_tensor is not None:
            rf_p = self.to_tensor(rf_p).float()
        return rf_p, int(info.TrueLabel)


class BkDatasetPatchTo3D(BkDatasetPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduce_op = '2d'
        self.to_tensor = transforms.ToTensor()
        if self.set_name == 'train':
            self.transform = iaa.Sequential([
                iaa.Fliplr(0.5),  # horizontal flips,
                # iaa.Affine(rotate=(-20, 20)),
                iaa.Rot90([0, 1, 3]),
                # iaa.GaussianBlur(sigma=(0.0, 1.0)),
                # iaa.Sharpen((0.0, 0.01)),
                iaa.CenterCropToFixedSize(*self.input_size),
            ], random_order=False)
        else:
            self.transform = iaa.CenterCropToFixedSize(*self.input_size)
        # self.signal_transform= tv.transforms.Compose([ToTensor1D()])


class BkDatasetPatchTeUS(BkDatasetPatch):
    def __init__(self, forced_input_size=(32, 32), *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.warning.default_params_dict = {'in_channel': 3, 'input_size': forced_input_size}
        # self.in_channel, self.input_size = 3, forced_input_size
        self.to_2d = To2D(self.in_channel, True)
        self.to_tensor = transforms.ToTensor()

        if self.set_name == 'train':
            self.transform_per_patch = iaa.Sequential([
                # iaa.Affine(rotate=(-20, 20)),
                iaa.CenterCropToFixedSize(*self.input_size),
            ], random_order=False)
        else:
            self.transform_per_patch = iaa.CenterCropToFixedSize(*self.input_size)
        if self.transform is None:
            if self.set_name == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(tuple([s * 10 for s in self.input_size]), padding=4),
                    # transforms.RandomCrop(tuple([s * 10 for s in self.input_size])),
                    transforms.ToTensor()
                ])

            #     self.transform = iaa.Sequential([
            #         # iaa.HorizontalFlip(),
            #         # iaa.CenterCropToFixedSize(*self.input_size),
            #     ], random_order=False)
            else:
                self.transform = transforms.ToTensor()
            # self.transform = iaa.Identity()

    def _transform(self, x, **kwargs):
        x = self.transform_per_patch(image=x)
        x = self.transform(self.to_2d(x))
        return x

    def __getitem__(self, idx):
        rf_p = self._get_patch(idx)
        rf_p = self._transform(rf_p)
        info = self.metadata.iloc[idx]
        return rf_p, int(info.TrueLabel)


class BkDatasetNeedleTeUS(BkDatasetPatchTeUS):
    def __init__(self, forced_input_size=(96, 96), *args, **kwargs):
        super().__init__(forced_input_size=forced_input_size, *args, **kwargs)
        self.transform_per_patch = iaa.Resize(self.input_size, interpolation='linear')

    def _get_patch(self, idx, *args, **kwargs):
        rf, roi, wp = self._load_numpy(idx)  # rf: HWF
        locs = np.argwhere((roi * wp) == 1)
        min_r, max_r = locs[:, 0].min(), locs[:, 0].max()
        min_c, max_c = locs[:, 1].min(), locs[:, 1].max()
        return rf[min_r:max_r, min_c: max_c]


class BkDatasetPatches(BkDatasetPatch):
    def __init__(self, *args, specified_metadata=None, extra_metadata=None, **kwargs):
        oversampling = kwargs['oversampling']
        kwargs['oversampling'] = False  # stop oversampling happened at the base class init
        super().__init__(*args, **kwargs)
        if self.patch_gen_config is None:
            self.patch_gen_config = {'batch_size': 32, 'workers': multiprocessing.cpu_count() - 1}

        if specified_metadata is None:
            self.prepare_patches()
            if oversampling:
                self.oversampling_minor_class()
        else:
            assert 'PatchId' in specified_metadata.columns
            self.metadata = specified_metadata
        if extra_metadata is not None:
            assert 'PatchId' in extra_metadata.columns
            self.metadata = pd.concat([self.metadata, extra_metadata])

    def __getitem__(self, idx):
        info = self.metadata.iloc[idx]
        rf_p = self._get_patch(idx, info.iloc[idx]['patch_locs']).astype('float32')
        rf_p = self._transform(self.reduce_dim(rf_p))
        rf_p = norm_robust(rf_p)
        if self.to_tensor is not None:
            rf_p = self.to_tensor(rf_p)
        return rf_p, int(info.TrueLabel)


class BkDatasetPatchesTo1D(BkDatasetPatchTo1D, BkDatasetPatches):
    """"""


class BkDatasetPatchesTo2D(BkDatasetPatchTo2D, BkDatasetPatches):
    """"""


class BkDatasetUnlabelledPatches2D(BkDatasetPatchTo2D, BkDatasetPatches):
    """"""

    def __init__(self, *args, random_crop=False, infomin=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_crop = random_crop
        self.infomin = infomin

    def _transform(self, x, roi=None, wp=None):
        return self.transform(x)

    def __getitem__(self, idx):
        rf_p = self._get_patch(idx).astype('float32')
        rf_p = self._transform(rf_p)
        rf_p = [norm_robust(_) for _ in rf_p]
        if self.to_tensor is not None:
            rf_p = [self.to_tensor(_).float() for _ in rf_p]
        return rf_p[0], rf_p[1]


class BkDatasetPatchesTo3D(BkDatasetPatchTo3D, BkDatasetPatches):
    """"""


class BkDatasetPatchesTeUS(BkDatasetPatchTeUS, BkDatasetPatches):
    """"""


class BkDatasetPatchesTo1DM(BkDatasetPatchTo1DM, BkDatasetPatches):
    """"""

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    # self.reduce_op = '1dm'

    """"""


class ParamWarnings:
    def __init__(self, model_name=None, default_params_dict=None):
        self.default_params_dict = default_params_dict
        self.model_name = model_name

    def inform(self, params_dict):
        suffix = ''
        if self.model_name is not None:
            suffix += f' for {self.model_name}'
        for k, v in self.default_params_dict.items():
            if k in params_dict.keys() and params_dict[k] != v:
                warnings.warn(f"{k}={params_dict[k]} is ignored.\n"
                              f"Using {k}={v} as this is the best setting{suffix}.\n")


def norm_0mean(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def norm_robust(x):
    return ((x - np.median(x)) /
            (np.percentile(x, 75, axis=1, keepdims=True) - np.percentile(x, 25, axis=1, keepdims=True)))


DATASETS = {
    'patch_1d': BkDatasetPatchTo1D,
    'patch_1dm': BkDatasetPatchTo1DM,
    'patch_2d': BkDatasetPatchTo2D,
    'patch_teus': BkDatasetPatchTeUS,
    'needle_teus': BkDatasetNeedleTeUS,
    'patches_1d': BkDatasetPatchesTo1D,
    'patches_1dm': BkDatasetPatchesTo1DM,
    'patches_2d': BkDatasetPatchesTo2D,
    'patches_unlabelled_2d': BkDatasetUnlabelledPatches2D,
    'patches_3d': BkDatasetPatchesTo3D,
    'patches_teus': BkDatasetPatchesTeUS,
    'rf_a': BkDatasetRfAatP,
    'rf_a_ssl': BkDatasetRfAatPforSSL,
}


def unit_test():
    ds = {}
    for set_name in ['train', 'val', 'test']:
        ds[set_name] = BkDatasetRfAatP(fold_idx=0, set_name=set_name, patch_dim=(32, 32, 0),
                                       in_channel=3, input_size=(256, 256),
                                       patch_gen_config={'batch_size': 64, 'workers': 12})

        dl = DataLoader(ds[set_name], shuffle=False, drop_last=False,
                        batch_size=32, num_workers=15)
        for x, y in (pbar := tqdm(dl, mininterval=10)):
            pbar.set_description(f'{dl.dataset.set_name} loop...', refresh=False)
            continue

        if isinstance(x, tuple) or isinstance(x, list):
            print([_x.shape for _x in x], y.shape)
        else:
            print(x.shape, y.shape)

class To2D:
    def __init__(self, in_channels=3, rescale=True):
        self.in_channels = in_channels
        self.rescale = rescale

    @staticmethod
    def reshape_to2d_image(patches):  # H W T
        m, n, t = patches.shape
        k = int(np.sqrt(t))
        image = np.concatenate(np.concatenate(patches.transpose(2, 0, 1).reshape(k, k, m, n), axis=1), axis=1)
        return image

    @staticmethod
    def rescale2d(img):
        # max_val = np.max(img)
        # min_val = np.min(img)
        # return (img - min_val) / float(max_val - min_val)
        return (img - float(img.mean())) / img.std()

    def __call__(self, core):
        img = self.reshape_to2d_image(core)
        if self.in_channels == 3:
            if self.rescale:
                # img = (self.rescale2d(img).astype('float32') * 255).astype('uint8')
                img = self.rescale2d(img.astype('float32'))
                # print(img.shape)
                # img = (self.normalize(img) * 255).astype('uint8')
                # img = (self.rescale2d(img.astype('float32')) * 255)
            img = img[:, :, np.newaxis]
            # img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img)  #, mode='RGB')
        else:
            assert self.in_channels == 1
            if self.rescale:
                # img = (img - img.mean()) / img.std()
                img = self.rescale2d(img.astype('float32'))
            img = Image.fromarray(img)
        return img


def scale_01(x: np.ndarray):
    return (x - x.min()) / (x.max() - x.min())


def plot_raw_rf(rf, contours: list = None, contours_c: list = ('r', 'b'), frm_idx=None, i=0, fig_size=(6, 6),
                target_shape=None, patch_masks=None, patch_idx=None, ax=None, convert_whole_rf=False):
    if convert_whole_rf:
        rf_frame = make_analytical(rf)[..., frm_idx] if frm_idx is not None else rf
        rf = rf[..., frm_idx] if frm_idx is not None else rf
    else:
        rf = rf[..., frm_idx] if frm_idx is not None else rf
        rf_frame = make_analytical(rf)

    line_widths = 1
    if target_shape is not None:
        rf_frame = cv2.resize(rf_frame, dsize=target_shape)
        line_widths = 0.2
    rf_frame = scale_01(rf_frame)

    if ax is None:
        plt.figure(i, figsize=fig_size, frameon=False)
        ax = plt.gca()

    rf_frame = np.flipud(rf_frame)
    if patch_masks is not None:
        patch_mask = patch_masks[patch_idx]
        com = np.argwhere(patch_mask == 1)
        rmin, rmax = min(com[:, 0]), max(com[:, 0])
        cmin, cmax = min(com[:, 1]), max(com[:, 1])
        selected_patch = rf[rmin:rmax, cmin:cmax]
        ref_shape = target_shape if target_shape is not None else rf_frame.shape
        patch_to_img_ratio = 3 if ref_shape[0] <= ref_shape[1] else 1.1
        upscale_ratio = (min(ref_shape) / patch_to_img_ratio) / min(selected_patch.shape)
        selected_patch = scale_01(cv2.resize(selected_patch,
                                             (0, 0),
                                             fx=upscale_ratio, fy=upscale_ratio
                                             ))
        start_r, start_c = 5, 5  # 5, rf_frame.shape[1] - selected_patch.shape[1] - 5

        h, w = selected_patch.shape
        rf_frame[start_r:start_r + h, start_c:start_c + w] = np.flipud(selected_patch)

        bounding_box = np.zeros(ref_shape)
        bounding_box[start_r:start_r + h, start_c:start_c + w] = 1
        plt.contour(bounding_box, colors='magenta', linewidths=line_widths * 2)

    ax.imshow(rf_frame, cmap='gray', vmin=rf_frame.min(), vmax=rf_frame.max())

    if contours is not None:
        for contour, contour_c in zip(contours, contours_c):
            if target_shape is not None:
                contour = cv2.resize(contour.astype('uint8'), dsize=target_shape, interpolation=cv2.INTER_NEAREST)
            plt.contour(np.flipud(contour), colors=contour_c)

    target_patch = None
    if patch_masks is not None:
        for i, patch_mask in enumerate(patch_masks):
            if target_shape is not None:
                patch_mask = cv2.resize(patch_mask.astype('uint8'), dsize=target_shape, interpolation=cv2.INTER_NEAREST)
            if patch_idx is not None and (i == patch_idx):
                target_patch = patch_mask
            plt.contour(np.flipud(patch_mask), colors='y', linewidths=line_widths)
        if target_patch is not None:
            plt.contour(np.flipud(target_patch), colors='magenta', linewidths=line_widths * 2)

    plt.axis('off')
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    return ax


def make_analytical(x: np.ndarray):
    return np.abs(hilbert(x)) ** 0.3


def get_dataset(fold_idx, set_name, transform=None):
    dataset_init = DATASETS['rf_a_ssl']

    ds = dataset_init(
        metadata_path=r'/projects/bk_pca/patches/data_splits/bk_Paul_splits/metadata.csv',  # hard-coded
        fold_idx=fold_idx,
        set_name=set_name.split('_')[0],  # so 'train_eval' will be 'train'
        in_channel=3,
        min_inv=0.0, min_gs=7,
        rf_save_dir='/projects/bk_pca/patches/',
        transform=transform,
    )
    return ds


if __name__ == '__main__':
    unit_test()
