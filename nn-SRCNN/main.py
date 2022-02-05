# %%
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models import SRCNN
from datasets import TrainDataset, EvalDataset
from torch.utils.data.dataloader import DataLoader
from utils import AverageMeter
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
criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': model.conv1.parameters()},
                        {'params': model.conv2.parameters()},
                        {'params': model.conv3.parameters(), 'lr': args.lr*0.1}],
                       lr=args.lr)
# %%
train_dataset = TrainDataset(args.train_file)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)
eval_dataset = EvalDataset(args.eval_file)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
best_epoch = 0
best_psnr = 0.0
# %%
for epoch in range(args.num_epochs):
    model.train()
    epoch_losses = AverageMeter()
