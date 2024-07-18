from typing import List, Dict, Tuple
import cv2
import numpy as np
import torch
import engine.transforms.functional as F

from xtdet.datasets import CLASSES
from xtdet.model.utils import center_of_mass


def show_result_ins(img: torch.Tensor,
                    result: Dict,
                    to_bgr_cfg:Dict,
                    score_thr=0.3,
                    sort_by_density=False,
                    ):

    img_show = img.permute(1, 2, 0).detach().cpu().numpy()
    img_show = np.ascontiguousarray(F.denormalize(img_show, mean=to_bgr_cfg['mean'], std=to_bgr_cfg['std']).astype(np.uint8))

    score, seg_label, cate_label = result['scores'], result['masks'], result['labels']

    cate_label = cate_label.cpu().numpy()
    score = score.detach().cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    colors = [
        np.random.randint(0, 256, 3, dtype=np.uint8)
        for _ in range(num_mask)
    ]

    for idx in range(num_mask):
        cur_mask = seg_label[idx, :, :]
        center_y, center_x = center_of_mass(cur_mask)
        if cur_mask.sum() == 0:
            continue
        cur_mask = cur_mask.cpu().numpy()
        color = colors[idx]
        img_show[..., 0] = img_show[..., 0] * cur_mask * 0.5 + color[0] * 0.5
        img_show[..., 1] = img_show[..., 1] * cur_mask * 0.5 + color[1] * 0.5
        img_show[..., 2] = img_show[..., 2] * cur_mask * 0.5 + color[2] * 0.5

        # 当前实例的类别
        cur_cate = cate_label[idx]
        label_text = CLASSES[cur_cate]
        cur_score = cate_score[idx]
        label_text += '|{:.02f}'.format(cur_score)

        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(img_show, label_text, vis_pos,
                   cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green

    return img_show
