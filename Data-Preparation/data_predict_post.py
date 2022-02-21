# %%
import tifffile as tf
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
print("OK!")
# %%


def preprocess(img, threshold):
    img_mask = cv.threshold(img, threshold, 255, cv.THRESH_TOZERO)[1]
    img = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=img_mask)
    return img.astype(np.uint8)


# %%
flag = False  # 是否使用 OTSU 计算阈值并过滤噪声
number = 2
plus = 10
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1+plus}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1+plus}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    tf.imwrite(f'../Data-Post-upsample/10x_predict/10x{i+plus+1}.tif', img10x)
    tf.imwrite(f'../Data-Post-upsample/20x_predict/20x{i+plus+1}.tif', img20x)

# %%
