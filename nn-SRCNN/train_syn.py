# %%
import os
import torch
import pandas as pd
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models import SRCNN
from datasets import TrainDataset, EvalDataset
from torch.utils.data.dataloader import DataLoader
from utils import AverageMeter, calc_psnr, calc_nqm, calc_ssim
from tqdm import tqdm
# print('Ready!')
# %%

if __name__ == '__main__':  # ! Must have this

    class Para(object):
        train_file = 'train_syn.h5'  # str
        eval_file = 'eval_syn.h5'  # str
        output_dir = './weight_output_syn'  # str
        lr = 1e-4  # float
        batch_size = 20  # int
        num_epochs = 100  # int
        num_workers = 0  # int
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
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    results = {'loss': [], 'psnr': [], 'ssim': [], 'nqm': []}

    for epoch in range(1, args.num_epochs+1):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset)-len(train_dataset) % args.batch_size)) as t:
            t.set_description(
                'epoch: {}/{}'.format(epoch, args.num_epochs))

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

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_ssim.update(calc_ssim(preds, labels), len(inputs))
            epoch_nqm.update(calc_nqm(preds, labels), len(inputs))

        print('eval psnr: {:.2f}, eval ssim: {:.2f}, eval nqm: {:.2f}'.format(
            epoch_psnr.avg, epoch_ssim.avg, epoch_nqm.avg))

        torch.save(model.state_dict(), os.path.join(
            args.output_dir, 'epoch_{}_lr_{:.8f}_psnr_{:.2f}_ssim{:.2f}_nqm{:.2f}.pth'.format(
                epoch, args.lr, epoch_psnr.avg, epoch_ssim.avg, epoch_nqm.avg)))

        results['loss'].append(epoch_losses.avg)
        results['psnr'].append(epoch_psnr.avg.cpu().squeeze(0).item())
        results['ssim'].append(epoch_ssim.avg.cpu().squeeze(0).item())
        results['nqm'].append(epoch_nqm.avg.cpu().squeeze(0).item())

# %%
data_frame = pd.DataFrame(
    data={'Loss': results['loss'],
          'PSNR': results['psnr'],
          'SSIM': results['ssim'],
          'NQM': results['nqm']
          }, index=range(1, epoch+1)
)
# %%
data_frame.to_csv('train_results_syn.csv', index_label='Epoch')
