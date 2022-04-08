# %%
import torch.backends.cudnn as cudnn
import torch
from models import EDSR
import torch.nn as nn
import torch.optim as optim
# %%
if __name__ == '__main__':
    class Para(object):
        train_file = 'train.h5'
        eval_file = 'eval.h5'
        output_dir = './weight_output'
        batch_size = 20
        num_epochs = 100
        lr = 1e-4
        step = 200
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
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay,
                           betas=(0.9, 0.999), eps=1e-8)
