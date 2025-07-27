from mmengine.registry import MODELS
from mmengine.model import BaseModule
from ..utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample, UpSample
from torch import nn
from einops import reduce, repeat
import torch


@MODELS.register_module()
class APESSegBackbone(BaseModule):
    def __init__(self, which_ds, init_cfg=None):
        super(APESSegBackbone, self).__init__(init_cfg)

        # === Modified to accept 6D input (XYZRGB) ===
        self.embedding = Embedding(input_dim=6)  # <-- Modified

        if which_ds == 'global':
            self.ds1 = GlobalDownSample(20480)
            self.ds2 = GlobalDownSample(10240)
            self.npts_ds = 20480
        elif which_ds == 'local':
            self.ds1 = LocalDownSample(20480)
            self.ds2 = LocalDownSample(10240)
            self.npts_ds = 20480
        else:
            raise NotImplementedError

        self.n2p_attention1 = N2PAttention()
        self.n2p_attention2 = N2PAttention()
        self.n2p_attention3 = N2PAttention()
        self.n2p_attention4 = N2PAttention()
        self.n2p_attention5 = N2PAttention()
        self.ups1 = UpSample()
        self.ups2 = UpSample()

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, shape_class):
        # Input: x = (B, 6, N) for XYZ + RGB
        N = x.shape[-1]

        tmp = self.embedding(x)                     # (B, 6, N) -> (B, 128, N)
        x1 = self.n2p_attention1(tmp)
        tmp = self.ds1(x1)
        x2 = self.n2p_attention2(tmp)
        tmp = self.ds2(x2)
        x3 = self.n2p_attention3(tmp)

        tmp = self.ups2(x2, x3)
        x2 = self.n2p_attention4(tmp)
        tmp = self.ups1(x1, x2)
        x1 = self.n2p_attention5(tmp)

        x = self.conv1(x1)
        x_max = reduce(x, 'B C N -> B C', 'max')
        x_avg = reduce(x, 'B C N -> B C', 'mean')
        x = torch.cat([x_max, x_avg], dim=1)        # (B, 2048)

        shape_class = self.conv2(shape_class).squeeze(-1)  # (B, 64)

        x = torch.cat([x, shape_class], dim=1)      # (B, 2112)
        x = repeat(x, 'B C -> B C N', N=N)

        if x.shape[-1] != x1.shape[-1]:
            min_n = min(x.shape[-1], x1.shape[-1])
            x = x[:, :, :min_n]
            x1 = x1[:, :, :min_n]

        x = torch.cat([x, x1], dim=1)               # (B, 2240, N)
        return x
