# %%
import os
import torch
import torch.backends.cudnn as cudnn
from datasets import TrainDataset, EvalDataset
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from tqdm import tqdm
from loss import GeneratorLoss
import torch.optim as optim
from torch.autograd import Variable
from utils import ssim
from math import log10
import pandas as pd
import torchvision.utils as utils
# %%
if __name__ == '__main__':
    class Para(object):
        train_file = 'train.h5'
        eval_file = 'eval.h5'
        output_dir = './weight_output'
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

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [],
               'd_score': [], 'g_score': [],
               'psnr': [], 'ssim': []}

    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_dataloader)
        running_results = {'batch_sizes': 0,
                           'd_loss': 0, 'g_loss': 0,
                           'd_score': 0, 'g_score': 0}
        netG.train()
        netD.train()

        for data, target in train_bar:
            # g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################

            real_img = target.to(device)
            z = data.to(device)

            fake_img = netG(z)
            fake_img = fake_img.to(device)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()

            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss

            netG.zero_grad()
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, args.num_epochs,
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()

        with torch.no_grad():
            eval_images = []
            eval_bar = tqdm(eval_dataloader)
            evaling_results = {'mse': 0, 'ssims': 0,
                               'psnr': 0, 'ssim': 0,
                               'batch_sizes': 0}

            for data, target in eval_bar:
                batch_size = data.size(0)
                evaling_results['batch_sizes'] += batch_size

                lr = data.to(device)
                hr = target.to(device)
                sr = netG(lr)

                batch_mse = ((sr-hr)**2).mean()
                evaling_results['mse'] += batch_mse*batch_size
                batch_ssim = ssim(sr, hr)
                evaling_results['ssims'] += batch_ssim*batch_size
                evaling_results['psnr'] = 10*log10(hr.max()**2/(
                    evaling_results['mse']/evaling_results['batch_sizes']))
                evaling_results['ssim'] = evaling_results['ssims'] / \
                    evaling_results['batch_sizes']
                eval_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f'
                    % (evaling_results['psnr'], evaling_results['ssim'])
                )
                eval_images.extend(lr.squeeze(0).squeeze(0),
                                   sr.squeeze(0).squeeze(0),
                                   hr.squeeze(0).squeeze(0))
            eval_images = torch.stack(eval_images)
            # ? torch.chunk()
            eval_save_bar = tqdm(eval_images, desc='[saving training results]')
            index = 1
            for image in eval_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(
                    image, args.output_dir + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        torch.save(netG.state_dict(), os.path.join(
            args.output_dir, 'netG_epoch_{}_{}.pth'.format(
                args.upscale_factor, epoch)))

        torch.save(netD.state_dict(), os.path.join(
            args.output_dir, 'netD_epoch_{}_{}.pth'.format(
                args.upscale_factor, epoch)))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(
            running_results['d_loss']/running_results['batch_sizes'])
        results['g_loss'].append(
            running_results['g_loss']/running_results['batch_sizes'])
        results['d_score'].append(
            running_results['d_score']/running_results['batch_sizes'])
        results['g_score'].append(
            running_results['g_score']/running_results['batch_sizes'])
        results['psnr'].append(evaling_results['psnr'])
        results['ssim'].append(evaling_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={
                    'Loss_D': results['d_loss'],
                    'Loss_G': results['g_loss'],
                    'Score_D': results['d_score'],
                    'Score_G': results['g_sclre'],
                    'PSNR': results['psnr'],
                    'SSIM': results['ssim']
                }, index=range(1, epoch+1)
            )
            data_frame.to_csv(args.output_dir+'srf_'+str(args.upscale_factor) +
                              '_train_results.csv', index_label='Epoch')
