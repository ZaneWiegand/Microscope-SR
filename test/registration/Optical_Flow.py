# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage.transform import warp
import tifffile as tf

# %%
new = tf.imread("new.tif", 0)
target = tf.imread("target.tif", 0)
# %%
v, u = optical_flow_tvl1(target, new)
# %%
nvec = 40  # Number of vectors to be displayed along each image dimension
nl, nc = target.shape
step = max(nl // nvec, nc // nvec)
# %%
y, x = np.mgrid[:nl:step, :nc:step]
u_ = u[::step, ::step]
v_ = v[::step, ::step]
# %%
norm = np.sqrt(u ** 2 + v ** 2)
plt.imshow(norm)
plt.quiver(x, y, u_, v_, color="r", units="dots", angles="xy", scale_units="xy")
plt.title("Optical flow magnitude and vector field")
plt.axis("off")
plt.show()
# %%
nr, nc = new.shape
row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
image_warp = warp(new, np.array([row_coords + v, col_coords + u]), mode="edge")
# %%
# build an RGB image with the unregistered sequence
unreg_im = np.zeros((nr, nc, 3))
unreg_im[..., 0] = new
unreg_im[..., 1] = target
unreg_im[..., 2] = target

# build an RGB image with the registered sequence
reg_im = np.zeros((nr, nc, 3))
reg_im[..., 0] = image_warp
reg_im[..., 1] = target
reg_im[..., 2] = target

# build an RGB image with the registered sequence
target_im = np.zeros((nr, nc, 3))
target_im[..., 0] = target
target_im[..., 1] = target
target_im[..., 2] = target
# %%


def transform(img):
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img


# %%
tf.imwrite("wrap_OpticalFlow.tif", transform(image_warp))
# plt.imsave('wrap.jpg',image_warp)
# %%
