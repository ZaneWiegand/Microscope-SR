# %%
import tifffile as tf
print("OK!")
# %%
hr_origin_dir = '../Data-Post-upsample/20x_origin'
lr_origin_dir = '../Data-Post-upsample/10x_origin'
hr_save_dir = '../Data-Post-upsample/20x_truth'
lr_save_dir = '../Data-Post-upsample/10x_predict'
pic_list = [11]
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
