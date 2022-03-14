# %%
import torch
from torch import nn
# %%


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        loss_srcnn = SRCNN()
        state_dict = loss_srcnn.state_dict()
        for n, p in torch.load('best.pth', map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        loss_net = nn.Sequential(*list(loss_srcnn.children())[:-2]).eval()
        for param in loss_net.parameters():
            param.requires_grad = False

        self.loss_net = loss_net
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss #! time consuming
        perception_loss = self.mse_loss(
            self.loss_net(out_images), self.loss_net(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        return 0.5 * image_loss + 0.001 * adversarial_loss + 0.5 * perception_loss


# %%
if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
