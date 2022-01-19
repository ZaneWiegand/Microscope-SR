# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
print('OK!')
# %%
# 0表示读取黑白图片，1表示读取彩色图片
new = plt.imread('knight.jpg')
# %%
plt.imshow(new, cmap='gray')
# %%
sift = cv.SIFT_create()
kp, descriptor = sift.detectAndCompute(new, None)
# %%
img = cv.drawKeypoints(
    new, kp, new, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
# %%
plt.imshow(img)
# %%
plt.imsave('knight_new.jpg', img)
# %%
