# %%
import os
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from datasets import TrainDataset, EvalDataset
from models import VDSR
from utils import AverageMeter, calc_psnr
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
# %%
if __name__ == '__main__':
    class Para(object):
        train_file = 'train.h5'
        eval_file = 'eval.h5'
        output_dir = './weight_output'
        batch_size = 20  # Training batch size
        num_epochs = 30  # Number of epochs to train for
        lr = 0.1  # Learning rate
        clip = 0.4  # Clipping Gradients
        momentum = 0.9  # Momentum (for optimizer)
        weight_decay = 1e-4  # Weight decay (for optimizer)
        step = 10  # Sets the learning rate to the initial LR decayed by momentum every n epochs
        num_workers = 0
        seed = 123

    args = Para()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    def adjust_learning_rate(Para, epoch):
        lr = Para.lr*(0.1**(epoch//Para.step))
        return lr

    model = VDSR().to(device)
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(1, args.num_epochs+1):
        lr = adjust_learning_rate(args, epoch-1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset)-len(train_dataset) % args.batch_size)) as t:
            t.set_description(
                'epoch: {}/{}, lr = {:.8f}'.format(epoch, args.num_epochs, optimizer.param_groups[0]["lr"]))
            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(
                    model.parameters(), args.clip)  # gradient explosion
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        torch.save(model.state_dict(), os.path.join(
            args.output_dir, 'epoch_{}_lr_{:.8f}_psnr_{:.2f}.pth'.format(epoch, lr, epoch_psnr.avg)))
