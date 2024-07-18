from typing import List

import torch
import torch.nn as nn

from engine.model import BUILD_NETWORK_REGISTRY

from ..utils import SELayer, ConvBNReLU

__all__ = [
    'DarkNet'
]

@BUILD_NETWORK_REGISTRY.register()
class DarkNet(nn.Module):
    def __init__(self, stages: List[List], out_indices=(0, 1, 2), stem_channels: int = 32):
        super().__init__()
        self.stem = ConvBNReLU(in_channels=3, out_channels=stem_channels, kernel_size=3, stride=2, padding=1)
        self.stages = self._generate_stages(stem_channels, stages)
        self.out_indices = out_indices
        return

    @staticmethod
    def _generate_stages(in_channels, stages_cfg: List[List]) -> nn.ModuleList:
        prev_out_channels = in_channels
        stages = nn.ModuleList()
        for stage in stages_cfg:
            stage_blocks = nn.Sequential()
            for i, (out_channels, dilations, group_width, stride, se_ratio) in enumerate(stage):
                d_block = DBlock(prev_out_channels, out_channels, dilations, group_width, stride, se_ratio)
                prev_out_channels = d_block.out_channels
                stage_blocks.add_module(f"{str(d_block)}#{i}", d_block)  # NOTE: {i} distinguishes blocks with same name
            stages.append(stage_blocks)
        return stages

    def forward(self, x):
        outputs = list()
        x_in = self.stem(x)
        for stage in self.stages:
            x_out = stage(x_in)
            outputs.append(x_out)
            x_in = x_out  # last stage out is next stage in
        return tuple([outputs[i] for i in self.out_indices])
