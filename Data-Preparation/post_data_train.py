# %%
import tifffile as tf
import cv2 as cv
print("OK!")
# %%
hr_origin_dir = '../Data-Post-upsample/20x_origin'
lr_origin_dir = '../Data-Post-upsample/10x_origin'
hr_save_dir = '../Data-Post-upsample/20x_train'
lr_save_dir = '../Data-Post-upsample/10x_train'
pic_list = [1, 2, 3, 4, 5, 6, 7, 8]
hr_size = 900
hr_stride_x = 500
hr_stride_y = 900
lr_size = hr_size//2
lr_stride_x = hr_stride_x//2
lr_stride_y = hr_stride_y//2
hr_data_num = 1
lr_data_num = 1
# %%
for pic_num in pic_list:
    img20x = tf.imread(f'{hr_origin_dir}/20x{pic_num}.tif')
    img10x = tf.imread(f'{lr_origin_dir}/10x{pic_num}.tif')
    img20x = img20x[60:-60, 80:-80]
    img10x = img10x[30:-30, 40:-40]
    for i in range(0, img20x.shape[1] - hr_size + 1, hr_stride_x):
        for j in range(0, img20x.shape[0] - hr_size + 1, hr_stride_y):
            hr = img20x[j:j + hr_size, i:i + hr_size]
            tf.imwrite(f'{hr_save_dir}/20x{hr_data_num}.tif', hr)
            hr_data_num += 1
    for i in range(0, img10x.shape[1] - lr_size + 1, lr_stride_x):
        for j in range(0, img10x.shape[0] - lr_size + 1, lr_stride_y):
            lr = img10x[j:j + lr_size, i:i + lr_size]
            tf.imwrite(f'{lr_save_dir}/10x{lr_data_num}.tif', lr)
            lr_data_num += 1
# %%
for pic_num in pic_list:
    for flip_f in [-1, 0, 1]:
        img20x = tf.imread(f'{hr_origin_dir}/20x{pic_num}.tif')
        img10x = tf.imread(f'{lr_origin_dir}/10x{pic_num}.tif')
        img20x = img20x[60:-60, 80:-80]
        img10x = img10x[30:-30, 40:-40]
        img20x = cv.flip(img20x, flip_f)
        img10x = cv.flip(img10x, flip_f)
        for i in range(0, img20x.shape[1] - hr_size + 1, hr_stride_x):
            for j in range(0, img20x.shape[0] - hr_size + 1, hr_stride_y):
                hr = img20x[j:j + hr_size, i:i + hr_size]
                tf.imwrite(f'{hr_save_dir}/20x{hr_data_num}.tif', hr)
                hr_data_num += 1
        for i in range(0, img10x.shape[1] - lr_size + 1, lr_stride_x):
            for j in range(0, img10x.shape[0] - lr_size + 1, lr_stride_y):
                lr = img10x[j:j + lr_size, i:i + lr_size]
                tf.imwrite(f'{lr_save_dir}/10x{lr_data_num}.tif', lr)
                lr_data_num += 1
# %%
for pic_num in pic_list:
    img20x = tf.imread(f'{hr_origin_dir}/20x{pic_num}.tif')
    img10x = tf.imread(f'{lr_origin_dir}/10x{pic_num}.tif')
    img20x = img20x[60:-60, 80:-80]
    img10x = img10x[30:-30, 40:-40]
    for theta in [30, 60, 90, 120, 150, 210, 240, 270, 300, 330]:
        rows_20x, cols_20x = img20x.shape
        rows_10x, cols_10x = img10x.shape
        M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), theta, 1)
        M_10x = cv.getRotationMatrix2D((cols_10x/2, rows_10x/2), theta, 1)
        img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
        img10x = cv.warpAffine(img10x, M_10x, (cols_10x, rows_10x))
        hr = img20x[450:1350, 750:1650]
        lr = img10x[225:675, 375:825]
        tf.imwrite(f'{hr_save_dir}/20x{hr_data_num}.tif', hr)
        hr_data_num += 1
        tf.imwrite(f'{lr_save_dir}/10x{lr_data_num}.tif', lr)
        lr_data_num += 1
