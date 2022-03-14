# %%
import torch
import torch.backends.cudnn as cudnn
from datasets import TrainDataset, EvalDataset
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from tqdm import tqdm
from loss import GeneratorLoss
import torch.optim as optim
from torch.autograd import Variable

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

            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()

            fake_img = netG(z)

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

# %%
