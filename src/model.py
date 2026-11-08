import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, ch=32, in_ch=1):
        super().__init__()
        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1), nn.ReLU(inplace=True))
        self.e1 = block(in_ch, ch)
        self.e2 = block(ch, 2 * ch)
        self.e3 = block(2 * ch, 4 * ch)
        self.p = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(4 * ch, 2 * ch, 2, stride=2)
        self.d2 = block(4 * ch, 2 * ch)
        self.u1 = nn.ConvTranspose2d(2 * ch, ch, 2, stride=2)
        self.d1 = block(2 * ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.p(a))
        c = self.e3(self.p(b))
        y = self.d2(torch.cat([self.u2(c), b], 1))
        y = self.d1(torch.cat([self.u1(y), a], 1))
        return self.out(y)
