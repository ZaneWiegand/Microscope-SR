# %%
import torch
import torch.nn as nn
# %%


class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, x)
        return output


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()

        self.conv_input = nn.Conv2d(
            in_channels=1, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False)
        self.residual = self.make_layer(Residual_Block, 12)
        self.conv_mid = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1, bias=False)
        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128*4,
                      kernel_size=3, padding=1, stride=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.conv_output = nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_input(x)
        res = self.residual(output)
        res = self.conv_mid(res)
        output = torch.add(res, output)
        output = self.upscale2x(output)
        output = self.conv_output(output)
        return output
