# %%
import tifffile as tf
import torch
import numpy as np
from utils import calc_ssim, calc_psnr, calc_nqm
import pandas as pd
# %%
if __name__ == '__main__':
    plus = 1
    number = 84
    print('real data:')
    results = {'psnr': [], 'ssim': [], 'nqm': []}
    for pic_number in range(number):
        lr_file = '../Data-Pre-upsample/10x_predict/10x{}.tif'.format(
            pic_number+plus)
        hr_file = '../Data-Pre-upsample/20x_truth/20x{}.tif'.format(
            pic_number+plus)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        image = tf.imread(lr_file)
        image = np.array(image, dtype=np.float32)

        input_img = image/255.
        input_img = torch.from_numpy(input_img).to(device)
        input_img = input_img.unsqueeze(0).unsqueeze(0)  # ? reason

        preds = input_img

        target = tf.imread(hr_file)
        target = np.array(target, dtype=np.float32)
        target = target/255.
        target = torch.from_numpy(target).to(device)
        target = target.unsqueeze(0).unsqueeze(0)

        psnr = calc_psnr(preds, target)
        ssim = calc_ssim(preds, target)
        nqm = calc_nqm(preds, target)
        print('PSNR: {:.2f}, SSIM: {:.2f}, NQM: {:.2f}'.format(
            psnr, ssim, nqm))

        results['psnr'].append(psnr.item())
        results['ssim'].append(ssim.item())
        results['nqm'].append(nqm.item())

    data_frame = pd.DataFrame(
        data={'PSNR': results['psnr'],
              'SSIM': results['ssim'],
              'NQM': results['nqm']
              }, index=range(1, number+1))
    data_frame.to_csv('test_results.csv', index_label='Epoch')

# %%
