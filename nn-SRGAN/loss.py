# %%
import torch
from torch import nn
# %%


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1-out_labels)

        # l1 Loss
        l1_loss = self.l1_loss(out_images, target_images)

        # mse Loss
        mse_loss = self.mse_loss(out_images, target_images)

        return l1_loss + 0.01*adversarial_loss
    # mse_loss + 0.01*adversarial_loss


# %%
if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
