# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import torch
from torch import nn


# Copy from `https://github.com/twtygqyy/pytorch-LapSRN/blob/master/lapsrn.py`
def get_upsample_filter(size: int) -> torch.Tensor:
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    bilinear_filter = torch.from_numpy((1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)).float()

    return bilinear_filter


class ConvLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvLayer, self).__init__()
        self.cl = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cl(x)

        return out


class LapSRN(nn.Module):
    def __init__(self):
        super(LapSRN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Scale 2
        laplacian_pyramid_conv1 = []
        for _ in range(10):
            laplacian_pyramid_conv1.append(ConvLayer(64))
        laplacian_pyramid_conv1.append(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
        laplacian_pyramid_conv1.append(nn.LeakyReLU(0.2, True))

        self.laplacian_pyramid_conv1 = nn.Sequential(*laplacian_pyramid_conv1)
        self.laplacian_pyramid_conv2 = nn.ConvTranspose2d(1, 1, (4, 4), (2, 2), (1, 1))
        self.laplacian_pyramid_conv3 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

        # Scale 4
        laplacian_pyramid_conv4 = []
        for _ in range(10):
            laplacian_pyramid_conv4.append(ConvLayer(64))
        laplacian_pyramid_conv4.append(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
        laplacian_pyramid_conv4.append(nn.LeakyReLU(0.2, True))

        self.laplacian_pyramid_conv4 = nn.Sequential(*laplacian_pyramid_conv4)
        self.laplacian_pyramid_conv5 = nn.ConvTranspose2d(1, 1, (4, 4), (2, 2), (1, 1))
        self.laplacian_pyramid_conv6 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

        # Scale 8
        laplacian_pyramid_conv7 = []
        for _ in range(10):
            laplacian_pyramid_conv7.append(ConvLayer(64))
        laplacian_pyramid_conv7.append(nn.ConvTranspose2d(64, 64, (4, 4), (2, 2), (1, 1)))
        laplacian_pyramid_conv7.append(nn.LeakyReLU(0.2, True))

        self.laplacian_pyramid_conv7 = nn.Sequential(*laplacian_pyramid_conv7)
        self.laplacian_pyramid_conv8 = nn.ConvTranspose2d(1, 1, (4, 4), (2, 2), (1, 1))
        self.laplacian_pyramid_conv9 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)

        # X2
        lpc1 = self.laplacian_pyramid_conv1(out)
        lpc2 = self.laplacian_pyramid_conv2(x)
        lpc3 = self.laplacian_pyramid_conv3(lpc1)
        out1 = lpc2 + lpc3
        # X4
        lpc4 = self.laplacian_pyramid_conv4(lpc1)
        lpc5 = self.laplacian_pyramid_conv5(out1)
        lpc6 = self.laplacian_pyramid_conv6(lpc4)
        out2 = lpc5 + lpc6
        # X8
        lpc7 = self.laplacian_pyramid_conv7(lpc4)
        lpc8 = self.laplacian_pyramid_conv8(out2)
        lpc9 = self.laplacian_pyramid_conv9(lpc7)
        out3 = lpc8 + lpc9

        return out1, out2, out3

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.ConvTranspose2d):
                c1, c2, h, w = module.weight.data.size()
                weight = get_upsample_filter(h)
                module.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if module.bias is not None:
                    module.bias.data.zero_()


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.sqrt(torch.pow((target - inputs), 2) + self.eps))

        return loss
