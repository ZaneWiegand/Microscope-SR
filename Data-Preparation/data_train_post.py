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
number = 7
j = 0
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    img10x = cv.flip(img10x, 1)
    img20x = cv.flip(img20x, 1)
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    img10x = cv.flip(img10x, 0)
    img20x = cv.flip(img20x, 0)
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    img10x = cv.flip(img10x, -1)
    img20x = cv.flip(img20x, -1)
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 0:600])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 0:1200])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[0:450, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[0:900, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[450:900, 600:1200])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[900:1800, 1200:2400])
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])

# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 30, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 30, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 60, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 60, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 90, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 90, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 120, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 120, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 150, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 150, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 210, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 210, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 240, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 240, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 270, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 270, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 300, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 300, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
# %%
for i in range(number):
    img10x = tf.imread(f'../Data-Post-upsample/10x_origin/10x{i+1}.tif')
    img20x = tf.imread(f'../Data-Post-upsample/20x_origin/20x{i+1}.tif')
    img10x = img10x[30:-30, 40:-40]
    img20x = img20x[60:-60, 80:-80]
    if flag:
        thresh10x, _ = cv.threshold(img10x, 0, 255, cv.THRESH_OTSU)
        thresh20x, _ = cv.threshold(img20x, 0, 255, cv.THRESH_OTSU)
        img10x = preprocess(img10x, thresh10x)
        img20x = preprocess(img20x, thresh20x)
    rows_10x, cols_10x = img10x.shape
    rows_20x, cols_20x = img20x.shape
    M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), 330, 1)
    M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), 330, 1)
    img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
    img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
    j = j+1
    tf.imwrite(
        f'../Data-Post-upsample/10x_train/10x{j}.tif', img10x[225:675, 300:900])
    tf.imwrite(
        f'../Data-Post-upsample/20x_train/20x{j}.tif', img20x[450:1350, 600:1800])
