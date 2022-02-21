# %%
import cv2 as cv
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")  # 忽略警告
print("OK!")
# %%


def transform(img):
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img


def preprocess(img, threshold):
    img_mask = cv.threshold(img, threshold, 255, cv.THRESH_TOZERO)[1]
    img = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=img_mask)
    return img


def create_image_block_stack(img, row, col):
    # 采用补零法，将图片分成row*col块
    imgsize_row, imgsize_col = img.shape[0], img.shape[1]
    blocksize_row = int(np.ceil(imgsize_row / row))
    blocksize_col = int(np.ceil(imgsize_col / col))
    paddle_img = np.zeros([blocksize_row * row, blocksize_col * col])
    paddle_img[0:imgsize_row, 0:imgsize_col] = img
    stack_img = np.zeros([row, col, blocksize_row, blocksize_col])
    for r in range(row):
        for c in range(col):
            stack_img[r, c, :, :] = paddle_img[
                r * blocksize_row: (r + 1) * blocksize_row,
                c * blocksize_col: (c + 1) * blocksize_col,
            ]

    return stack_img.astype(np.uint8)


def display_blocks(divide_image):
    m, n = divide_image.shape[0], divide_image.shape[1]
    plt.figure(figsize=(10, 10))
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            plt.imshow(divide_image[i, j], vmin=0, vmax=255)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def stitch_block(stack_img, origin):
    # stack_img是四维向量，前两维储存位置，后两维储存图像块
    # origin用于获取原图尺寸
    num_row, num_col, block_row, block_col = stack_img.shape
    full_img = np.zeros([num_row * block_row, num_col * block_col])
    for i in range(num_row):
        for j in range(num_col):
            full_img[
                i * block_row: (i + 1) * block_row, j * block_col: (j + 1) * block_col
            ] = stack_img[i, j, :, :]
    full_img = full_img[0: origin.shape[0], 0: origin.shape[1]]
    return full_img


# %%
test_number = 2
pic10x = tf.imread(
    "/Users/zanewiegand/代码/python/Microscope-Super-Resolution/Registration/Raw-Data/10X/region{}.tif".format(
        test_number
    )
)
pic20x = tf.imread(
    "/Users/zanewiegand/代码/python/Microscope-Super-Resolution/Registration/Raw-Data/20X/region{}.tif".format(
        test_number
    )
)
pic10x = pic10x[:, :, 1]
pic20x = pic20x[:, :, 1]
# %%
pre_threshold = 20
pic10x = preprocess(transform(pic10x), pre_threshold)
pic20x = preprocess(transform(pic20x), pre_threshold)
# %%
pic10x_r, pic10x_c = pic10x.shape
pic10x_ex = cv.resize(
    pic10x, [pic10x_c * 2, pic10x_r * 2], interpolation=cv.INTER_LINEAR
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
new = pic10x_ex_cut
target = pic20x
# %%


def rigid_registration(obj_stack, ref_stack, target_all, flag=False):
    num_row, num_col, nr, nc = ref_stack.shape
    ans_stack = np.zeros_like(ref_stack)
    for i in range(num_row):
        for j in range(num_col):
            new = obj_stack[i, j, :, :]
            target = ref_stack[i, j, :, :]
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(new, None)
            kp2, des2 = sift.detectAndCompute(target, None)
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])
            if flag:
                img = cv.drawMatchesKnn(
                    new,
                    kp1,
                    target,
                    kp2,
                    good_matches,
                    None,
                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
            good_matches = np.squeeze(good_matches)
            # 物体特征点坐标
            ref_matched_kpts = np.float32(
                [kp1[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            # 场景特征点坐标
            sensed_matched_kpts = np.float32(
                [kp2[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            # 方法采用RANSAC计算投影矩阵，阈值设为5.0，即误差的2范数超过5.0，视为局外点

            #
            H, status = cv.findHomography(
                ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0
            )
            warp_img = cv.warpPerspective(
                new,
                H,
                (target.shape[1], target.shape[0]),
                borderMode=cv.BORDER_REPLICATE,
            )
            ans_stack[i, j, :, :] = warp_img
    origin_img = stitch_block(obj_stack, target_all)
    return origin_img.astype(np.uint8)


# %%
block_row = 1
block_col = 1
new_block_stack = create_image_block_stack(new, block_row, block_col)
target_block_stack = create_image_block_stack(target, block_row, block_col)
warp_img = rigid_registration(new_block_stack, target_block_stack, target)
plt.figure(figsize=(9, 9))
plt.imshow(warp_img)
plt.axis("off")
plt.show()
# %%


def calculate_MSD(target, warp):
    warp = warp.astype(np.uint8)
    target = target.astype(np.uint8)
    diff_pic = warp - target
    diff = np.mean(diff_pic * diff_pic)
    return diff


def calculate_MI(target, warp):
    warp = warp.astype(np.uint8)
    target = target.astype(np.uint8)
    target = np.reshape(target, -1)
    warp = np.reshape(warp, -1)
    size = warp.shape[-1]
    px = np.histogram(warp, 256, (0, 255))[0] / size
    py = np.histogram(target, 256, (0, 255))[0] / size
    hx = -np.sum(px * np.log(px + 1e-8))
    hy = -np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(warp, target, 256, [[0, 255], [0, 255]])[0]
    hxy /= 1.0 * size
    hxy = -np.sum(hxy * np.log(hxy + 1e-8))

    MI = hx + hy - hxy
    return MI


def calculate_NMI(target, warp):
    warp = warp.astype(np.uint8)
    target = target.astype(np.uint8)
    target = np.reshape(target, -1)
    warp = np.reshape(warp, -1)
    size = warp.shape[-1]
    px = np.histogram(warp, 256, (0, 255))[0] / size
    py = np.histogram(target, 256, (0, 255))[0] / size
    hx = -np.sum(px * np.log(px + 1e-8))
    hy = -np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(warp, target, 256, [[0, 255], [0, 255]])[0]
    hxy /= 1.0 * size
    hxy = -np.sum(hxy * np.log(hxy + 1e-8))

    NMI = (hx + hy) / hxy
    return NMI


def calculate_NCC(target, warp):
    warp = warp.astype(np.uint8)
    target = target.astype(np.uint8)
    up = np.sum((target - np.mean(target)) * (warp - np.mean(warp)))
    down1 = np.sqrt(np.sum((target - np.mean(target))
                    * (target - np.mean(target))))
    down2 = np.sqrt(np.sum((warp - np.mean(warp)) * (warp - np.mean(warp))))
    return up / (down1 * down2)


# %%
# print(calculate_MI(target, first_warp_img))
print(calculate_MI(target, warp_img))
# %%
# print(calculate_NCC(target, first_warp_img))
print(calculate_NCC(target, warp_img))
# %%
# print(calculate_NMI(target, first_warp_img))
print(calculate_NMI(target, warp_img))
# %%
# print(calculate_MSD(target, first_warp_img))
print(calculate_MSD(target, warp_img))
# %%
plt.imshow(target - warp_img)
# %%
ans = target - warp_img
