# %%
import cv2 as cv
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
from Match_PIPE import preprocess, global_registration, create_image_block_stack, blocks_registration, compare_unreg_reg, calculate_MI, calculate_MSD, calculate_NCC, calculate_NMI
import warnings
warnings.filterwarnings("ignore")  # 忽略警告
print("OK!")
# %%
#!############################################################################################
# *###########################################################################################
# ?###########################################################################################
# *###########################################################################################
#!############################################################################################
#! 荧光主程序开始
test_number = 1
pic10x = tf.imread(
    "./Raw-Data/10X/region{}.tif".format(
        test_number
    )
)
pic20x = tf.imread(
    "./Raw-Data/20X/region{}.tif".format(
        test_number
    )
)
pic10x = pic10x[:, :, 1]
pic20x = pic20x[:, :, 1]
# %%
# ? 合适阈值
pre_threshold = 0
pic10x = preprocess(pic10x, pre_threshold)
pic20x = preprocess(pic20x, pre_threshold)

# %%
# 插值pic1，变为原来的4倍
pic10x_r, pic10x_c = pic10x.shape
pic10x_ex = cv.resize(
    pic10x, [pic10x_c * 2, pic10x_r * 2], interpolation=cv.INTER_CUBIC
)
method = cv.TM_SQDIFF_NORMED
result = cv.matchTemplate(pic10x_ex, pic20x, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
t1 = min_loc
pic20x_r, pic20x_c = pic20x.shape
br = (t1[0] + pic20x_r, t1[1] + pic20x_c)
pic10x_ex_cut = pic10x_ex[
    min_loc[0]: min_loc[0] + pic20x_r, min_loc[1]: min_loc[1] + pic20x_c
]
# %%
new = pic10x_ex_cut
target = pic20x
# %%
# * method = "SIFT","ORB","KAZE","AKAZE","BRISK","nCCM"
global_method = "KAZE"
warp_img = global_registration(new, target, global_method, True)
first_warp_img = warp_img
# %%
block_row_ini = block_row = 1
block_col_ini = block_col = 1
target_block_stack = create_image_block_stack(target, block_row, block_col)
# display_blocks(target_block_stack)
# %%
threshold = 0.25
layer = 16
layer_state = np.ones(layer)
warp_img_layer = np.zeros([warp_img.shape[0], warp_img.shape[1], layer])
target_img_layer = np.zeros([target.shape[0], target.shape[1], layer])
target = target.astype(np.float64)
warp_img = warp_img.astype(np.float64)
pic20x = pic20x.astype(np.float64)
method = cv.INTER_LINEAR
# %%
for i in range(layer):
    count = 0
    rtmap = np.zeros_like(target)
    ctmap = np.zeros_like(target)
    warp_img_layer[:, :, i] = warp_img  # i=0,是SIFT矫正的底层图
    target_img_layer[:, :, i] = target
    rmax_shift = cmax_shift = 1
    while ~(rmax_shift <= threshold and cmax_shift <= threshold):
        target = (target + warp_img + pic20x) / 3
        warp_block_stack = create_image_block_stack(
            warp_img, block_row, block_col)
        warp_img, rmax_shift, cmax_shift, rtmap, ctmap = blocks_registration(
            warp_block_stack, target_block_stack, target, method, True
        )
        print("第{}层位移: (r = {}, c = {})".format(i + 1, rmax_shift, cmax_shift))
        count += 1
        rtmap += rtmap
        ctmap += ctmap
        if (count >= 4) and (rtmap.any() >= 1 or ctmap.any() >= 1):
            layer_state[i] = 0
            warp_img = warp_img_layer[:, :, i]
            target = target_img_layer[:, :, i]
            print("第{}层位移有误，从第{}层结果重新划分".format(i + 1, i))
            break
    block_row = block_row + 1
    block_col = block_col + 1

    target_block_stack = create_image_block_stack(target, block_row, block_col)
# %%
compare_unreg_reg(new, target, first_warp_img, global_method, 0, False)
compare_unreg_reg(new, target, warp_img, global_method + "-PIPE", 0, False)
plt.figure(figsize=(10, 10))
# plt.imshow(20*np.log10(np.abs(first_warp_img - warp_img)))
plt.imshow(np.abs(first_warp_img - warp_img))
plt.axis("off")
plt.show()
# tf.imwrite("{}-PIPE.tif".format(global_method),np.abs(first_warp_img - warp_img))
target = pic20x
# %%
# target = preprocess(target, 14)
# first_warp_img = preprocess(first_warp_img, 14)
# warp_img = preprocess(warp_img, 14)
# %%
print(calculate_MI(target, first_warp_img))
print(calculate_MI(target, warp_img))
# %%
print(calculate_NCC(target, first_warp_img))
print(calculate_NCC(target, warp_img))
# %%
print(calculate_NMI(target, first_warp_img))
print(calculate_NMI(target, warp_img))
# %%
print(calculate_MSD(target, first_warp_img))
print(calculate_MSD(target, warp_img))
# %%
tup = [(i, layer_state[i]) for i in range(len(layer_state))]
k = np.max([j for j, n in tup if n == 1])
print("max layer: {}".format(k + 1))
for j, n in tup:
    if n == 1:
        print(
            "layer: {} , block size: {}".format(
                j + 1,
                (
                    int(np.ceil(target.shape[0] / (j + block_row_ini))),
                    int(np.ceil(target.shape[1] / (j + block_col_ini))),
                ),
            )
        )

# %%
if True:
    if method == cv.INTER_CUBIC:
        methods = "CUBIC"
    elif method == cv.INTER_LINEAR:
        methods = "LINEAR"
    elif method == cv.INTER_NEAREST:
        methods = "NEAREST"
    else:
        methods = "ERROR"
    tf.imwrite(
        "./Test-Result/test{}_pic_10x_{}_{}_{}_{}_{}divide.tif".format(
            test_number, pre_threshold, global_method, methods, threshold, k + 1
        ),
        warp_img.astype(np.uint8),
    )
    tf.imwrite(
        "./Test-Result/test{}_pic10x.tif".format(test_number),
        pic10x_ex_cut.astype(np.uint8),
    )
    tf.imwrite(
        "./Test-Result/test{}_pic_20x.tif".format(
            test_number), pic20x.astype(np.uint8)
    )
#! 荧光主程序结束
#!############################################################################################
# *###########################################################################################
# ?###########################################################################################
# *###########################################################################################
#!############################################################################################
# %%
if False:
    def display_blocks(divide_image):
        m, n = divide_image.shape[0], divide_image.shape[1]
        plt.figure(figsize=(8, 6))
        for i in range(m):
            for j in range(n):
                plt.subplot(m, n, i * n + j + 1)
                plt.imshow(divide_image[i, j], vmin=0, vmax=255)
                plt.axis("off")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.08, hspace=0.08)
        plt.show()

    warp_block_stack = create_image_block_stack(warp_img, 12, 12)
    target_block_stack = create_image_block_stack(target, 12, 12)
    display_blocks(warp_block_stack)
    display_blocks(target_block_stack)
# %%
