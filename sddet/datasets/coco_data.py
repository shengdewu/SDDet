from typing import List, Optional, Dict, Tuple, Union
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from engine.data.build import BUILD_DATASET_REGISTRY
from engine.data.dataset import EngineDataSet
import engine.transforms.functional as F


__all__ = [
    'ConcatDataset',
    'CocoDataset'
]


@BUILD_DATASET_REGISTRY.register()
class ConcatDataset(Dataset):
    def __init__(self,
                 transformer: List,
                 ann_info:Union[List[Tuple], Tuple],
                 filter_empty_gt=True,
                 use_cate=None,
                 select_nums=-1,
                 poly2mask=True
                 ):
        super(ConcatDataset, self).__init__()
        self.datasets = list()
        if not isinstance(ann_info, List):
            ann_info = [ann_info]

        for ann in ann_info:
            data_path, ann_path = ann
            self.datasets.append(CocoDataset(data_path, ann_path, transformer, filter_empty_gt, use_cate, select_nums, poly2mask))
        self.cumulative_sizes = self.cumsum(self.datasets)
        return

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @staticmethod
    def find_dataset_idx(a, x, lo=0, hi=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = self.find_dataset_idx(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


@BUILD_DATASET_REGISTRY.register()
class CocoDataset(EngineDataSet):

    def __init__(self,
                 data_path,
                 ann_path,
                 transformer: List,
                 filter_empty_gt=True,
                 use_cate=None,
                 select_nums=-1,
                 poly2mask=True):
        super(CocoDataset, self).__init__(transformer)

        self.poly2mask = poly2mask
        self.data_path = data_path
        self.ann_path = ann_path
        self.filter_empty_gt = filter_empty_gt
        self.use_cate = use_cate
        if isinstance(self.use_cate, str):
            self.use_cate = [use_cate]

        self.coco = COCO(self.ann_path)
        self.img_infos, self.cat2label, self.cat_ids, self.img_ids = self.load_annotations()
        self.resort_label()

        # filter images too small
        valid_inds = self._filter_imgs()
        self.img_infos = [self.img_infos[i] for i in valid_inds]
        # set group flag for the sampler
        self._set_group_flag()

        if 0 < select_nums < len(self.img_infos):
            self.img_infos = random.sample(self.img_infos, k=select_nums)
        return

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self):
        cat_ids = self.coco.getCatIds()
        cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(cat_ids)
        }
        img_ids = self.coco.getImgIds()

        img_infos = []
        for i in img_ids:
            info = self.coco.loadImgs([i])[0]
            img_name = info['file_name']
            info['img_path'] = f'{self.data_path}/{img_name}'
            img_infos.append(info)
        return img_infos, cat2label, cat_ids, img_ids

    def resort_label(self):
        if self.use_cate is not None:
            use_ids = [self.CLASSES.index(cate) + 1 for cate in self.use_cate]
            new_cat2label = dict()

            start_ids = 1
            for cat_id, idx in self.cat2label.items():
                if idx in use_ids:
                    new_cat2label[cat_id] = start_ids
                    start_ids += 1

            self.cat2label = new_cat2label
        return

    def get_use_cate(self, ann_info):
        if self.use_cate is not None:
            use_ids = [self.CLASSES.index(cate) + 1 for cate in self.use_cate]
            ann_info = [ann for ann in ann_info if ann['category_id'] in use_ids]
        if len(ann_info) == 0:
            return None
        return ann_info

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        ann_info = self.get_use_cate(ann_info)
        if ann_info is None:
            return None
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['file_name'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        while True:
            data = self.prepare_train_img(idx)
            if data is not None:
                break
            idx = self._rand_another(idx)
        return data

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(mask_ann, img_h, img_w)
            rle = mask_utils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = mask_utils.decode(rle)
        return mask

    def pipeline(self, img_info, ann_info:Optional[Dict]):
        results = dict()

        img = cv2.imread(img_info['img_path'], cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) # inplace

        results['img'] = img
        results['img_fields'] = ['img']
        results['color_fields'] = ['img']
        results['gt_bboxes'] = ann_info['bboxes']
        results['bbox_fields'] = ['gt_bboxes']
        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')

        h, w = img_info['height'], img_info['width']
        gt_masks = ann_info['masks']
        results['interpolation'] = dict()
        gt_masks_keys = list()
        for i, mask in enumerate(gt_masks):
            if self.poly2mask:
                mask = self._poly2mask(mask, h, w)
            key = f'gt_masks_{i}'
            results[key] = mask
            results['img_fields'].append(key)
            results['interpolation'][key] = 'INTER_NEAREST'
            gt_masks_keys.append(key)

        results['gt_labels'] = ann_info['labels']

        results = self.data_pipeline(results)

        collect_results = dict(
            img=F.to_tensor(results['img']),
            gt_labels = torch.from_numpy(results['gt_labels']),
            gt_bboxes = torch.from_numpy(results['gt_bboxes']),
            img_meta=dict(ori_shape=results['ori_shape'],
                          img_shape=results['img_shape'],
                          scale=results['scale'],
                          keep_ratio=results['keep_ratio'],
                          pad_offset=results.get('pad_offset', (0, 0, 0, 0)),
                          img_norm_cfg=results['img_norm_cfg'])
        )

        if len(gt_masks_keys) > 0:
            collect_results['gt_masks'] = torch.from_numpy(np.stack([results[key] for key in gt_masks_keys], axis=0))

        return collect_results

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        if ann_info is None:
            return None
        return self.pipeline(img_info, ann_info)
