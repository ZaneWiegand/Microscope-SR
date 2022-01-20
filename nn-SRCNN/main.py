# %%
import torch
import torch.nn as nn
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
print('Ready!')
# %%
im = tf.imread('../Data/10x/10x1.tif')
plt.imshow(im)

# %%
