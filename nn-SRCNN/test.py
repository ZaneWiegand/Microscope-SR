# %%
import torch
import torch.backends.cudnn as cudnn
from models import SRCNN
# %%


class Para(object):
    weights_file = ''
    image_file = ''


# %%
arg = Para()
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
