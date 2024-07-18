from engine.trainer.trainer import BaseTrainer
import logging
from engine.trainer.build import BUILD_TRAINER_REGISTRY
import torch
import cv2

from .visualization import show_result_ins

from torch.utils.data.dataloader import default_collate
import numpy as np


def pad(img_list, max_size):
    batch_shape = [len(img_list)] + list(img_list[0].shape[:-2]) + max_size
    pad_images = img_list[0].new_full(batch_shape, 0.0)
    for img, pad_image in zip(img_list, pad_images):
        pad_image[..., 0: img.shape[-2], 0: img.shape[-1]].copy_(img)
    return pad_images


def collate(batch):

    return_collect = dict()
    for k in batch[0]:
        collect = [d[k] for d in batch]
        if k in ['img']:
            size = [(im.shape[-2], im.shape[-1]) for im in collect]
            size = [torch.as_tensor(x) for x in size]
            max_size = torch.stack(size).max(0).values.numpy().tolist()
            return_collect[k] = pad(collect, max_size)
        else:
            return_collect[k] = collect

    return return_collect


@BUILD_TRAINER_REGISTRY.register()
class DetTrainer(BaseTrainer):

    def __init__(self, cfg):
        super(DetTrainer, self).__init__(cfg)
        return

    def set_collate_fn(self, cfg):
        self.collate_train_fn = collate
        self.collate_valid_fn = collate
        return

    def after_loop(self):
        self.model.disable_train()

        for i, batch in enumerate(self.test_data_loader):
            with torch.no_grad():
                result = self.model(batch)
            img = batch['img']
            img_norm_cfg = result['img_norm_cfg']
            for b in range(len(result)):
                show_img = show_result_ins(img[b], result[b], img_norm_cfg)
                cv2.imwrite(f'{self.output}/{i}-{b}', show_img)
        return
