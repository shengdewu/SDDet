from engine.model.build import BUILD_NETWORK_REGISTRY
from engine.model.build import build_network
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Union, List, Tuple
from xtdet.model.utils.accuracy import accuracy

__all__ = [
    'SingleStageInstanceSegmentor'
]


@BUILD_NETWORK_REGISTRY.register()
class SingleStageInstanceSegmentor(nn.Module):
    def __init__(self, backbone: Dict,
                 header: Dict,
                 necker: Optional[Dict] = None):
        super(SingleStageInstanceSegmentor, self).__init__()
        self.backbone = build_network(backbone)
        self.head = build_network(header)
        self.with_neck = False
        if necker is not None:
            self.necker = build_network(necker)
            self.with_neck = True
        return

    @property
    def model_name(self):
        return self.encoder.__class__.__name__ + '-' + self.decoder.__class__.__name__

    def extract_feature(self, img: Tensor):
        x = self.backbone(img)
        if self.with_neck:
            x = self.necker(x)
        return x

    def forward_train(self, img: Tensor, gt_labels, gt_masks, gt_bboxes=None) -> Dict[str, Tensor]:
        x = self.extract_feature(img)
        return self.head.forward_train(x, gt_labels, gt_masks, gt_bboxes)

    def forward(self, img: Tensor) -> List[Dict[str, Tensor]]:
        x = self.extract_feature(img)
        seg_pred = self.head.forward_test(x, img.shape[2:])
        return seg_pred

    def preparate_deploy(self):
        self.head.preparate_deploy()
        return
