# %%
from torch import nn
# %%


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)

        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                  groups=1, bias=True, padding_mode='zeros', device=None, dType=None)
