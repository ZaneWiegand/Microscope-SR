# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.transform import warp
import tifffile as tf
import scipy.optimize as opt
# %%


def transform(img):
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img


def preprocess(img):
    img_mask = cv.threshold(img, np.min(img), np.max(img), cv.THRESH_OTSU)[1]
    img = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=img_mask)
    return img


# %%
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

    return stack_img


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
def fft_CMM(f, g):
    # 根据 Fast Normalized Cross-Correlation J. P. Lewis Industrial Light & Magic
    fmean = np.mean(f)
    gmean = np.mean(g)
    f = f - fmean
    g = g - gmean
    F = np.fft.fft2(f)
    G = np.fft.fft2(g)
    xx = F * np.conj(G)
    CMMmap = np.fft.fftshift(np.fft.ifft2(xx))
    return CMMmap.real


# %%
def PPMCC(obj, ref, r, c):
    row, col = ref.shape
    if r >= 0 and c >= 0:
        obj_sub = obj[0: row - r, 0: col - c]
        ref_sub = ref[r:, c:]
    elif r >= 0 and c < 0:
        c = -c
        obj_sub = obj[0: row - r, c:]
        ref_sub = ref[r:, 0: col - c]
    elif r < 0 and c >= 0:
        r = -r
        obj_sub = obj[r:, 0: col - c]
        ref_sub = ref[0: row - r, c:]
    else:
        r = -r
        c = -c
        obj_sub = obj[r:, c:]
        ref_sub = ref[0: row - r, 0: col - c]
    if (
        np.sum((obj_sub - np.mean(obj_sub)) ** 2) == 0
        or np.sum((ref_sub - np.mean(ref_sub)) ** 2) == 0
    ):
        return 1
    return np.sum(
        (obj_sub - np.mean(obj_sub)) * (ref_sub - np.mean(ref_sub))
    ) / np.sqrt(
        np.sum((obj_sub - np.mean(obj_sub)) ** 2)
        * np.sum((ref_sub - np.mean(ref_sub)) ** 2)
    )


def n_CMM(CMM, PPMCC_max, PPMCC_min):
    if np.max(CMM) == np.min(CMM):
        if np.max(CMM) == 0:
            return CMM
        else:
            nCMM = CMM / np.max(CMM)
    else:
        nCMM = (CMM - np.min(CMM)) / (np.max(CMM) - np.min(CMM))
        nCMM = nCMM * (PPMCC_max - PPMCC_min) + PPMCC_min
    return nCMM


def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = amplitude * np.exp(
        -((x - xo) ** 2 / (2 * sigma_x ** 2) + (y - yo) ** 2 / (2 * sigma_y ** 2))
    )
    return g


def compare_unreg_reg(obj, ref, warp, flag=False):
    # flag = True 则保存对比图片
    nr, nc = ref.shape
    seq_un = np.zeros((nr, nc, 3)).astype(np.uint8)
    seq_un[..., 0] = transform(obj)
    seq_un[..., 1] = transform(ref)
    seq_un[..., 2] = transform(ref)
    seq_re = np.zeros((nr, nc, 3)).astype(np.uint8)
    seq_re[..., 0] = transform(warp)
    seq_re[..., 1] = transform(ref)
    seq_re[..., 2] = transform(ref)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(seq_un)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(seq_re)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    if flag:
        tf.imwrite("seq_un.tif", seq_un)
        tf.imwrite("seq_re.tif", seq_re)


# %%
i = 3
pic10x = tf.imread(
    "/Users/zanewiegand/学习/毕业设计/毕业设计数据/荧光/template_matching/10X/region{}.tif".format(
        i)
)
pic20x = tf.imread(
    "/Users/zanewiegand/学习/毕业设计/毕业设计数据/荧光/template_matching/20X/region{}.tif".format(
        i)
)
pic10x = pic10x[:, :, 1]
pic20x = pic20x[:, :, 1]
pic10x = preprocess(transform(pic10x))
pic20x = preprocess(transform(pic20x))
# 插值pic1，变为原来的4倍
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
# %%
new = pic10x_ex_cut
target = pic20x
block_row = 6
block_col = 6
obj_stack = create_image_block_stack(new, block_row, block_col)
ref_stack = create_image_block_stack(target, block_row, block_col)
num_row, num_col, nr, nc = ref_stack.shape
CMM_stack = np.zeros_like(ref_stack)
nCMM_stack = np.zeros_like(ref_stack)
r_shift_stack = np.zeros([num_row, num_col, 1, 1])
c_shift_stack = np.zeros([num_row, num_col, 1, 1])
pcov_stack = np.zeros([num_row, num_col, 5, 5])
# %%
for i in range(num_row):
    for j in range(num_col):
        obj_copy = obj_stack[i, j, :, :]
        ref_copy = ref_stack[i, j, :, :]
        CMM_stack[i, j, :, :] = fft_CMM(obj_copy, ref_copy)
        maxloc = divmod(np.argmax(CMM_stack[i, j, :, :]), nc)
        r_max = int(np.round(maxloc[0] - nr / 2))
        c_max = int(np.round(maxloc[1] - nc / 2))
        minloc = divmod(np.argmin(CMM_stack[i, j, :, :]), nr)
        r_min = int(np.round(minloc[0] - nr / 2))
        c_min = int(np.round(minloc[1] - nc / 2))
        PPMCC_max = PPMCC(
            obj_copy,
            ref_copy,
            r_max,
            c_max,
        )
        PPMCC_min = PPMCC(
            obj_copy,
            ref_copy,
            r_min,
            c_min,
        )
        nCMM_stack[i, j, :, :] = n_CMM(
            CMM_stack[i, j, :, :], PPMCC_max, PPMCC_min)
        # 针对nCMM的分布进行不同处理
        if np.abs(r_max) > int(nr / 4) or np.abs(c_max) > int(nc / 4):
            r_shift_stack[i, j, 0, 0] = 0
            c_shift_stack[i, j, 0, 0] = 0
        else:
            nCMM_stack_single = nCMM_stack[i, j, :, :]
            y = np.linspace(0, nc - 1, nc)
            x = np.linspace(0, nr - 1, nr)
            x, y = np.meshgrid(x, y)
            # ? sigma的合理估算
            sigma_x = np.mean(np.var(nCMM_stack_single, axis=1))
            sigma_y = np.mean(np.var(nCMM_stack_single, axis=0))
            initial_guess = (
                PPMCC_max,
                nc / 2 + c_max,
                nr / 2 + r_max,
                sigma_x,
                sigma_y,
            )
            popt, pcov = opt.curve_fit(
                twoD_Gaussian,
                np.vstack((x.ravel(), y.ravel())),
                nCMM_stack_single.ravel(),
                p0=initial_guess,
            )
            pcov_stack[i, j, :, :] = pcov
            r_shift_stack[i, j, 0, 0] = popt[2] - nr / 2
            c_shift_stack[i, j, 0, 0] = popt[1] - nc / 2

r_translation_map = stitch_block(r_shift_stack, np.zeros([num_row, num_col]))
c_translation_map = stitch_block(c_shift_stack, np.zeros([num_row, num_col]))

# %%
# TODO 奇异值点的剔除与化零
r_left = np.mean(r_translation_map) - 3 * np.std(r_translation_map)
r_right = np.mean(r_translation_map) + 3 * np.std(r_translation_map)
c_left = np.mean(c_translation_map) - 3 * np.std(c_translation_map)
c_right = np.mean(c_translation_map) + 3 * np.std(c_translation_map)
r_translation_map[r_translation_map > r_right] = 0
r_translation_map[r_translation_map < r_left] = 0
c_translation_map[c_translation_map > c_right] = 0
c_translation_map[c_translation_map < c_left] = 0
# %%
target_shape = (target.shape[1], target.shape[0])
r_interpolated_translation_map = cv.resize(
    r_translation_map, target_shape, interpolation=method
)
c_interpolated_translation_map = cv.resize(
    c_translation_map, target_shape, interpolation=method
)
# TODO 奇异值点的剔除与化零

r_left = np.mean(r_interpolated_translation_map) - 3 * np.std(
    r_interpolated_translation_map
)
r_right = np.mean(r_interpolated_translation_map) + 3 * np.std(
    r_interpolated_translation_map
)
c_left = np.mean(c_interpolated_translation_map) - 3 * np.std(
    c_interpolated_translation_map
)
c_right = np.mean(c_interpolated_translation_map) + 3 * np.std(
    c_interpolated_translation_map
)
r_interpolated_translation_map[r_interpolated_translation_map > r_right] = 0
r_interpolated_translation_map[r_interpolated_translation_map < r_left] = 0
# r_interpolated_translation_map[r_interpolated_translation_map > nr / 2] = 0
# r_interpolated_translation_map[r_interpolated_translation_map < -nr / 2] = 0
c_interpolated_translation_map[c_interpolated_translation_map > c_right] = 0
c_interpolated_translation_map[c_interpolated_translation_map < c_left] = 0
# c_interpolated_translation_map[c_interpolated_translation_map > nc / 2] = 0
# c_interpolated_translation_map[c_interpolated_translation_map < -nc / 2] = 0
# %%
origin_img = stitch_block(obj_stack, target)
origin_img[:, -1] = 100
row_coords, col_coords = np.meshgrid(
    np.arange(target.shape[0]), np.arange(target.shape[1]), indexing="ij"
)
warp_img = warp(
    origin_img,
    np.array(
        [
            row_coords + r_interpolated_translation_map,
            col_coords + c_interpolated_translation_map,
        ]
    ),
    mode="edge",
)
# %%
nCMM = stitch_block(nCMM_stack, target)
plt.figure(figsize=(10, 10))
plt.imshow(nCMM)
plt.title("Stitched nCMM")
plt.axis("off")
plt.show()
# %%
plt.figure(figsize=(5, 5))
step = 1
y, x = np.mgrid[
    : r_translation_map.shape[0]: step, : r_translation_map.shape[1]: step
]
r_ = r_translation_map[::step, ::step]
c_ = c_translation_map[::step, ::step]
plt.quiver(x, y, -c_, -r_, color="r", units="dots",
           angles="xy", scale_units="xy")
plt.title("Block Vector Field")
plt.axis("off")
plt.show()
# %%
nvec = int(np.min(target.shape) / 50)
step = max(target.shape[0] // nvec, target.shape[1] // nvec)
y, x = np.mgrid[: target.shape[0]: step, : target.shape[1]: step]
r_ = r_interpolated_translation_map[::step, ::step]
c_ = c_interpolated_translation_map[::step, ::step]
plt.figure(figsize=(10, 10))
# plt.imshow(warp_img)
plt.quiver(x, y, -c_, -r_, color="r", units="dots",
           angles="xy", scale_units="xy")
plt.title("Interpolated Vector Field")
plt.axis("off")
plt.show()
# %%
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(origin_img)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(warp_img)
plt.axis("off")
plt.show()
