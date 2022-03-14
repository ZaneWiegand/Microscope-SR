# %%
import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from datasets import TrainDataset, EvalDataset
from torch.utils.data import DataLoader
from models import Generator, Discriminator
# %%
if __name__ == '__main__':
    class Para(object):
        train_file = 'train.h5'
        eval_file = 'eval.h5'
        upscale_factor = 2
        batch_size = 32
        num_epochs = 100
        num_workers = 0
        seed = 123

    args = Para()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    train_dataset = TrainDataset(args.train_file)
    eval_dataset = EvalDataset(args.eval_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        num_workers=args.num_workers,
        batch_size=1,
        shuffle=True)

    netG = Generator(args.upscale_factor)
    print('# generator parameters:', sum(param.numel()
          for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel()
          for param in netD.parameters()))
