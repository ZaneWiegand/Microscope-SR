# %%
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch
from models import EDSR
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from datasets import TrainDataset, EvalDataset
from utils import calc_psnr, calc_ssim, calc_nqm, AverageMeter
from tqdm import tqdm
# %%
if __name__ == '__main__':
    class Para(object):
        train_file = 'train.h5'
        eval_file = 'eval.h5'
        output_dir = './weight_output'
        batch_size = 20
        num_epochs = 100
        lr = 1e-4
        step = 25
        momentum = 0.9
        weight_decay = 1e-4
        num_workers = 0
        seed = 123

    args = Para()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    def adjust_learning_rate(Para, epoch):
        lr = Para.lr*(0.1**(epoch//Para.step))
        return lr

    model = EDSR().to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay,
                           betas=(0.9, 0.999), eps=1e-8)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    results = {'loss': [], 'psnr': [], 'ssim': [], 'nqm': []}

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
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_nqm = AverageMeter()

        eval_bar = tqdm(eval_dataloader)
        for data in eval_bar:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(preds, labels), len(inputs))
            epoch_nqm.update(calc_nqm(preds, labels), len(inputs))

            eval_bar.set_description(
                desc='[LR images --> SR images] PSNR: %.4f dB SSIM: %.4f NQM: %.4f dB'
                % (epoch_psnr.avg, epoch_ssim.avg, epoch_nqm.avg)
            )

        torch.save(model.state_dict(), os.path.join(
            args.output_dir, 'epoch_{}_lr_{:.8f}_psnr_{:.2f}_ssim{:.2f}_nqm{:.2f}.pth'.format(
                epoch, lr, epoch_psnr.avg, epoch_ssim.avg, epoch_nqm.avg)))

        results['loss'].append(epoch_losses.avg)
        results['psnr'].append(epoch_psnr.avg.cpu().squeeze(0).item())
        results['ssim'].append(epoch_ssim.avg.cpu().squeeze(0).item())
        results['nqm'].append(epoch_nqm.avg.cpu().squeeze(0).item())

# %%
data_frame = pd.DataFrame(
    data={'Loss': results['loss'], }, index=range(1, epoch+1))
# %%
data_frame.to_csv('train_results.csv', index_label='Epoch')
