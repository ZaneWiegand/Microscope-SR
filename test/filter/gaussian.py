# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import convolve2d
print('OK!')
# %%
x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
y_filter = x_filter.T
# %%
pic = cv.imread('knight.jpg', 0)
newx = convolve2d(pic, x_filter)
newy = convolve2d(pic, y_filter)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(newx, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(newy, cmap='gray')
plt.tight_layout()
# %%
template = np.zeros([9, 9])
xcenter = (template.shape[0]-1)/2
ycenter = (template.shape[1]-1)/2
# %%


def gaussian(x, y, xcenter, ycenter):
    sigma = 1.2
    return 1/(2*np.pi*sigma*sigma)*np.exp((-(x-xcenter)**2-(y-ycenter)**2)/(2*sigma*sigma))


# %%
for i in range(template.shape[0]):
    for j in range(template.shape[1]):
        template[i, j] = gaussian(i, j, xcenter, ycenter)
# %%
plt.imshow(template, cmap='gray')
# %%
newx = convolve2d(template, x_filter, mode='same')
newx = convolve2d(newx, x_filter, mode='same')
plt.imshow(newx, cmap='gray')
# %%
newy = convolve2d(template, y_filter, mode='same')
newy = convolve2d(newy, y_filter, mode='same')
plt.imshow(newy, cmap='gray')
# %%
newx = convolve2d(template, x_filter, mode='same')
newxy = convolve2d(newx, y_filter, mode='same')
plt.imshow(newxy, cmap='gray')
# %%
