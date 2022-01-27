# %%
import torch
import torch.backends.cudnn as cudnn
from models import SRCNN
print('Ready!')
# %%


class Para(object):
    train_file = '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/nn-SRCNN/train.h5'  # str
    eval_file = '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/nn-SRCNN/eval.h5'  # str
    outputs_dir = '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/nn-SRCNN'  # str
    lr = 1e-4  # float
    batch_size = 20  # int
    num_epochs = 100  # int
    num_workers = 8  # int
    seed = 123  # int


# %%
args = Para()
# %%
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
# %%
model = SRCNN().to(device)
