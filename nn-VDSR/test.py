# %%
import torch
import torch.backends.cudnn as cudnn
from models import VDSR
import tifffile as tf
import numpy as np
from utils import calc_psnr
# %%
if __name__ == '__main__':
    weights_file = './weight_output/best.pth'
    plus = 11
    number = 2
    # %%
    for pic_number in range(number):
        image_file = '../Data-Pre-upsample/10x_predict/10x{}.tif'.format(
            pic_number+plus)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = VDSR().to(device)

        state_dict = model.state_dict()  # shallow copy
        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        model.eval()  # eval mode
        image = tf.imread(image_file)
        image = np.array(image, dtype=np.float32)

        input_img = image/255.
        input_img = torch.from_numpy(input_img).to(device)  # ? reason
        input_img = input_img.unsqueeze(0).unsqueeze(0)  # ? reason

        with torch.no_grad():
            preds = model(input_img).clamp(0.0, 1.0)

        psnr = calc_psnr(input_img, preds)
        print('PSNR: {:.2f}'.format(psnr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(
            0).squeeze(0).astype(np.uint8)  # ? reason
        tf.imsave('./pic_output/10x_out{}.tif'.format(pic_number+plus), preds)
