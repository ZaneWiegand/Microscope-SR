# %%
import tifffile as tf
import cv2 as cv
import numpy as np
print("OK!")
# %%


def create_gauss_kernel(kernel_size, sigma, x0, y0, dr):
    rx = dr*kernel_size[1]/2
    ry = dr*kernel_size[0]/2
    X = np.linspace(x0-rx, x0+rx, kernel_size[1])
    Y = np.linspace(y0-ry, y0+ry, kernel_size[0])
    x, y = np.meshgrid(X, Y)
    gauss = 1/(2*np.pi*sigma**2)*np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))
    return gauss


def PSF_process(img, kernel):
    img_f = np.fft.fftshift(np.fft.fft2(img))
    PSF_f = np.fft.fftshift(np.fft.fft2(kernel))
    out_f = img_f*PSF_f
    out = np.fft.fftshift(np.fft.ifft2(out_f))
    return np.abs(out).astype(np.uint8)


def add_Gauss_noise(img, mean, sigma):
    img = img/255
    noise = np.random.normal(mean, sigma, img.shape)
    out = img+noise
    out = np.clip(out, 0, 1)*255
    return out.astype(np.uint8)


def add_Poisson_noise(img, PEAK):
    img = img/255
    out = np.random.poisson(img*255*PEAK)/(255*PEAK)
    out = np.clip(out, 0, 1)*255
    return out.astype(np.uint8)


def downsample(img, F):
    img = cv.resize(img, (img.shape[0]//F, img.shape[1]//F), cv.INTER_CUBIC)
    return img


def degradation(img, resize_flag):
    # first degradation blur [0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3]
    # add noise Gauss [0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002] or Poisson [2,3,4,5,6,7,8,9]
    # resize (if necessary)
    # second degradation blur [0.8,0.9,1.0,1.1]
    # add noise Gauss [0.005,0.004,0.003,0.002] or Poisson [6,7,8,9]
    blur = np.random.choice([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    PSF = create_gauss_kernel(img.shape, blur, 0, 0, 1)
    img = PSF_process(img, PSF)

    if resize_flag:
        img = downsample(img, 2)

    if np.random.choice([0, 1]):
        sigma = np.random.choice(
            [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002])
        img = add_Gauss_noise(img, 0, sigma)
    else:
        PEAK = np.random.choice([2, 3, 4, 5, 6, 7, 8, 9])
        img = add_Poisson_noise(img, PEAK)

    blur = np.random.choice([0.8, 0.9, 1.0, 1.1])
    PSF = create_gauss_kernel(img.shape, blur, 0, 0, 1)
    img = PSF_process(img, PSF)

    if np.random.choice([0, 1]):
        sigma = np.random.choice([0.005, 0.004, 0.003, 0.002])
        img = add_Gauss_noise(img, 0, sigma)
    else:
        PEAK = np.random.choice([6, 7, 8, 9])
        img = add_Poisson_noise(img, PEAK)

    return img


# %%
hr_origin_dir = '../Data-Post-upsample/20x_origin'
hr_save_dir = '../Data-Post-upsample/20x_train_syn'
lr_save_dir = '../Data-Post-upsample/10x_train_syn'
pic_list = [1, 2, 3, 4, 5, 6, 7, 8]
hr_size = 900
hr_stride_x = 500
hr_stride_y = 900
hr_data_num = 1
resize_flag = True
lr_data_num = 1
# %%
for pic_num in pic_list:
    img20x = tf.imread(f'{hr_origin_dir}/20x{pic_num}.tif')
    img20x = img20x[60:-60, 80:-80]
    for i in range(0, img20x.shape[1] - hr_size + 1, hr_stride_x):
        for j in range(0, img20x.shape[0] - hr_size + 1, hr_stride_y):
            hr = img20x[j:j + hr_size, i:i + hr_size]
            tf.imwrite(f'{hr_save_dir}/20x{hr_data_num}.tif', hr)
            hr_data_num += 1
            lr = degradation(hr, resize_flag)
            tf.imwrite(f'{lr_save_dir}/10x{lr_data_num}.tif', lr)
            lr_data_num += 1
# %%
for pic_num in pic_list:
    for flip_f in [-1, 0, 1]:
        img20x = tf.imread(f'{hr_origin_dir}/20x{pic_num}.tif')
        img20x = img20x[60:-60, 80:-80]
        img20x = cv.flip(img20x, flip_f)
        for i in range(0, img20x.shape[1] - hr_size + 1, hr_stride_x):
            for j in range(0, img20x.shape[0] - hr_size + 1, hr_stride_y):
                hr = img20x[j:j + hr_size, i:i + hr_size]
                tf.imwrite(f'{hr_save_dir}/20x{hr_data_num}.tif', hr)
                hr_data_num += 1
                lr = degradation(hr, resize_flag)
                tf.imwrite(f'{lr_save_dir}/10x{lr_data_num}.tif', lr)
                lr_data_num += 1
# %%
for pic_num in pic_list:
    img20x = tf.imread(f'{hr_origin_dir}/20x{pic_num}.tif')
    img20x = img20x[60:-60, 80:-80]
    for theta in [30, 60, 90, 120, 150, 210, 240, 270, 300, 330]:
        rows_20x, cols_20x = img20x.shape
        M_20x = cv.getRotationMatrix2D((cols_20x/2, rows_20x/2), theta, 1)
        img20x = cv.warpAffine(img20x, M_20x, (cols_20x, rows_20x))
        hr = img20x[450:1350, 750:1650]
        tf.imwrite(f'{hr_save_dir}/20x{hr_data_num}.tif', hr)
        hr_data_num += 1
        lr = degradation(hr, resize_flag)
        tf.imwrite(f'{lr_save_dir}/10x{lr_data_num}.tif', lr)
        lr_data_num += 1
