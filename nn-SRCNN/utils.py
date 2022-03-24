# %%
import torch
# %%


def calc_psnr(sr, hr):
    return 10.*torch.log10(hr.max()**2/torch.mean((hr-sr)**2))
# %%


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # 返回的val是每个Batch的MSELoss
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
