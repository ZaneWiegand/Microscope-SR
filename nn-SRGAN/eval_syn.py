# %%
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from models import Generator
from datasets import EvalDataset
from torch.utils.data.dataloader import DataLoader
from utils import AverageMeter, calc_psnr, calc_nqm, calc_ssim
from tqdm import tqdm
# %%

if __name__ == '__main__':  # ! Must have this

    class Para(object):
        eval_file = 'eval_syn.h5'  # str
        weight_dir = './weight_output_syn'  # str
        num_epochs = 100  # int
        num_workers = 0  # int
        seed = 123  # int
        upscale_factor = 2
    # %%
    args = Para()
    # %%
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model = Generator(args.upscale_factor).to(device)
    # %%
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    results = {'psnr': [], 'ssim': [], 'nqm': []}
    # %%
    for epoch in range(1, args.num_epochs+1):
        weights_file = '{}/netG_F2_epoch_{}.pth'.format(
            args.weight_dir, epoch)
        state_dict = model.state_dict()
        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

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
                desc='[epoch: %.1d/%.1d LR images --> SR images] PSNR: %.4f dB SSIM: %.4f NQM: %.4f dB'
                % (epoch, args.num_epochs, epoch_psnr.avg, epoch_ssim.avg, epoch_nqm.avg)
            )

        results['psnr'].append(epoch_psnr.avg.cpu().squeeze(0).item())
        results['ssim'].append(epoch_ssim.avg.cpu().squeeze(0).item())
        results['nqm'].append(epoch_nqm.avg.cpu().squeeze(0).item())

# %%
data_frame = pd.DataFrame(
    data={'PSNR': results['psnr'],
          'SSIM': results['ssim'],
          'NQM': results['nqm']
          }, index=range(1, epoch+1)
)
# %%
data_frame.to_csv('eval_results_syn.csv', index_label='Epoch')
