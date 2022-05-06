# %%
import torch
import torch.backends.cudnn as cudnn
from models import VDSR
import tifffile as tf
import numpy as np
from utils import calc_ssim, calc_psnr, calc_nqm
# %%
if __name__ == '__main__':
    weights_file = './weight_output/epoch_100.pth'
    upscale_factor = 2
    plus = 1
    number = 8
    print('real data:')
    for pic_number in range(number):
        lr_file = '../Data-Pre-upsample/10x_predict/10x{}.tif'.format(
            pic_number+plus)
        hr_file = '../Data-Pre-upsample/20x_truth/20x{}.tif'.format(
            pic_number+plus)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = VDSR().to(device)

        state_dict = model.state_dict()
        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        model.eval()

        image = tf.imread(lr_file)
        image = np.array(image, dtype=np.float32)

        input_img = image/255.
        input_img = torch.from_numpy(input_img).to(device)  # ? reason
        input_img = input_img.unsqueeze(0).unsqueeze(0)  # ? reason

        target = tf.imread(hr_file)
        target = np.array(target, dtype=np.float32)
        target = target/255.
        target = torch.from_numpy(target).to(device)
        target = target.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(input_img).clamp(0.0, 1.0)

        psnr = calc_psnr(preds, target)
        ssim = calc_ssim(preds, target)
        nqm = calc_nqm(preds, target)
        print('PSNR: {:.2f}, SSIM: {:.2f}, NQM: {:.2f}'.format(
            psnr, ssim, nqm))

        preds = preds.mul(255.0).cpu().numpy().squeeze(
            0).squeeze(0).astype(np.uint8)  # ? reason
        tf.imwrite('./pic_output/10x_out{}.tif'.format(pic_number+plus), preds)

    ################################################################################################

    weights_file = './weight_output_syn/epoch_100.pth'
    upscale_factor = 2
    plus = 1
    number = 8
    print('syn data:')
    for pic_number in range(number):
        lr_file = '../Data-Pre-upsample/10x_predict/10x{}.tif'.format(
            pic_number+plus)
        hr_file = '../Data-Pre-upsample/20x_truth/20x{}.tif'.format(
            pic_number+plus)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = VDSR().to(device)

        state_dict = model.state_dict()
        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        model.eval()

        image = tf.imread(lr_file)
        image = np.array(image, dtype=np.float32)

        input_img = image/255.
        input_img = torch.from_numpy(input_img).to(device)  # ? reason
        input_img = input_img.unsqueeze(0).unsqueeze(0)  # ? reason

        target = tf.imread(hr_file)
        target = np.array(target, dtype=np.float32)
        target = target/255.
        target = torch.from_numpy(target).to(device)
        target = target.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(input_img).clamp(0.0, 1.0)

        psnr = calc_psnr(preds, target)
        ssim = calc_ssim(preds, target)
        nqm = calc_nqm(preds, target)
        print('PSNR: {:.2f}, SSIM: {:.2f}, NQM: {:.2f}'.format(
            psnr, ssim, nqm))

        preds = preds.mul(255.0).cpu().numpy().squeeze(
            0).squeeze(0).astype(np.uint8)  # ? reason
        tf.imwrite(
            './pic_output_syn/syn_10x_out{}.tif'.format(pic_number+plus), preds)
