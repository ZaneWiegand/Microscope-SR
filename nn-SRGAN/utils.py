from math import exp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
from torch.fft import fft2, ifft2, fftshift
from zmq import device
# %%


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calc_ssim(sr, hr, window_size=11, size_average=True):
    (_, channel, _, _) = sr.size()
    window = create_window(window_size, channel)

    if sr.is_cuda:
        window = window.cuda(sr.get_device())
    window = window.type_as(sr)

    return _ssim(sr, hr, window, window_size, channel, size_average)


def calc_psnr(sr, hr):
    return 10.*torch.log10(hr.max()**2/torch.mean((hr-sr)**2))


def ctf(f_r):
    f_r = torch.Tensor([f_r])
    y = 1/(200*2.6*(0.0192+0.114*f_r)*torch.exp(-(0.114*f_r)**1.1))
    return y


def cmaskn(c, ci, a, ai, i):
    cx = deepcopy(c)
    cix = deepcopy(ci)
    cix[torch.abs(cix) > 1] = 1
    ct = ctf(i).to(c.get_device())
    T = ct*(.86*((cx/ct)-1)+.3)
    ai[(abs(cix-cx)-T) < 0] = a[(abs(cix-cx)-T) < 0]
    return ai


def gthresh(x, T, z):
    T = T.to(x.get_device())
    z[torch.abs(x) < T] = 0
    return z


def calc_nqm(sr, hr, VA=np.pi/3):
    device = sr.get_device()
    _, _, row, col = sr.shape
    X = torch.linspace(-row/2+0.5, row/2-0.5, row)
    Y = torch.linspace(-col/2+0.5, col/2-0.5, col)
    x, y = torch.meshgrid(X, Y)
    plane = (x+1j*y)
    r = torch.abs(plane)

    pi = np.pi
    G_0 = 0.5*(1+torch.cos(pi*torch.log2((r+2)*(r+2 >= 1) *
                                         (r+2 <= 4)+4*(~((r+2 <= 4)*(r+2 >= 1))))-pi))

    G_1 = 0.5 * \
        (1+torch.cos(pi*torch.log2(r*((r >= 1)*(r <= 4))+4*(~((r >= 1)*(r <= 4))))-pi))

    G_2 = 0.5 * \
        (1+torch.cos(pi*torch.log2(r*((r >= 2)*(r <= 8))+.5*(~((r >= 2) * (r <= 8))))))

    G_3 = 0.5*(1+torch.cos(pi*torch.log2(r*((r >= 4)*(r <= 16)) +
                                         4*(~((r >= 4)*(r <= 16))))-pi))

    G_4 = 0.5*(1+torch.cos(pi*torch.log2(r*((r >= 8) *
                                            (r <= 32))+.5*(~((r >= 8) * (r <= 32))))))

    G_5 = 0.5*(1+torch.cos(pi*torch.log2(r*((r >= 16)*(r <= 64)) +
                                         4*(~((r >= 16)*(r <= 64))))-pi))
    GS_0 = fftshift(G_0).to(device)
    GS_1 = fftshift(G_1).to(device)
    GS_2 = fftshift(G_2).to(device)
    GS_3 = fftshift(G_3).to(device)
    GS_4 = fftshift(G_4).to(device)
    GS_5 = fftshift(G_5).to(device)

    FO = fft2(sr).to(device)
    FI = fft2(hr).to(device)

    L_0 = GS_0*FO
    LI_0 = GS_0*FI

    l_0 = torch.real(ifft2(L_0))
    li_0 = torch.real(ifft2(LI_0))

    A_1 = GS_1*FO
    AI_1 = GS_1*FI

    a_1 = torch.real(ifft2(A_1))
    ai_1 = torch.real(ifft2(AI_1))

    A_2 = GS_2*FO
    AI_2 = GS_2*FI

    a_2 = torch.real(ifft2(A_2))
    ai_2 = torch.real(ifft2(AI_2))

    A_3 = GS_3*FO
    AI_3 = GS_3*FI

    a_3 = torch.real(ifft2(A_3))
    ai_3 = torch.real(ifft2(AI_3))

    A_4 = GS_4*FO
    AI_4 = GS_4*FI

    a_4 = torch.real(ifft2(A_4))
    ai_4 = torch.real(ifft2(AI_4))

    A_5 = GS_5*FO
    AI_5 = GS_5*FI

    a_5 = torch.real(ifft2(A_5))
    ai_5 = torch.real(ifft2(AI_5))

    c1 = a_1/l_0
    c2 = a_2/(l_0+a_1)
    c3 = a_3/(l_0+a_1+a_2)
    c4 = a_4/(l_0+a_1+a_2+a_3)
    c5 = a_5/(l_0+a_1+a_2+a_3+a_4)

    ci1 = ai_1/li_0
    ci2 = ai_2/(li_0+ai_1)
    ci3 = ai_3/(li_0+ai_1+ai_2)
    ci4 = ai_4/(li_0+ai_1+ai_2+ai_3)
    ci5 = ai_5/(li_0+ai_1+ai_2+ai_3+ai_4)

    d1 = ctf(2/VA)
    d2 = ctf(4/VA)
    d3 = ctf(8/VA)
    d4 = ctf(16/VA)
    d5 = ctf(32/VA)

    ai_1 = cmaskn(c1, ci1, a_1, ai_1, 1)
    ai_2 = cmaskn(c2, ci2, a_2, ai_2, 2)
    ai_3 = cmaskn(c3, ci3, a_3, ai_3, 3)
    ai_4 = cmaskn(c4, ci4, a_4, ai_4, 4)
    ai_5 = cmaskn(c5, ci5, a_5, ai_5, 5)

    l0 = l_0
    li0 = li_0
    a1 = gthresh(c1, d1, a_1)
    ai1 = gthresh(ci1, d1, ai_1)
    a2 = gthresh(c2, d2, a_2)
    ai2 = gthresh(ci2, d2, ai_2)
    a3 = gthresh(c3, d3, a_3)
    ai3 = gthresh(ci3, d3, ai_3)
    a4 = gthresh(c4, d4, a_4)
    ai4 = gthresh(ci4, d4, ai_4)
    a5 = gthresh(c5, d5, a_5)
    ai5 = gthresh(ci5, d5, ai_5)

    Os = l0+a1+a2+a3+a4+a5
    Is = li0+ai1+ai2+ai3+ai4+ai5

    A = torch.sum(Os**2)
    square_err = (Os-Is)*(Os-Is)
    B = torch.sum(square_err)
    nqm_value = 10*torch.log10(A/B)
    return nqm_value


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
