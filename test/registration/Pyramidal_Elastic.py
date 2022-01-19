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
        # 这里的数据分布不波动的PPMCC返回值设为0
        return 0
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
def blocks_registration(obj_stack, ref_stack, target, flag=False):
    num_row, num_col, nr, nc = ref_stack.shape
    CMM_stack = np.zeros_like(ref_stack)
    nCMM_stack = np.zeros_like(ref_stack)
    r_shift_stack = np.zeros([num_row, num_col, 1, 1])
    c_shift_stack = np.zeros([num_row, num_col, 1, 1])
    pcov_stack = np.zeros([num_row, num_col, 5, 5])
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

    # TODO r_shift_stack与c_shift_stack中奇异值点的剔除化零
    r_translation_map = stitch_block(
        r_shift_stack, np.zeros([num_row, num_col]))
    c_translation_map = stitch_block(
        c_shift_stack, np.zeros([num_row, num_col]))
    target_shape = (target.shape[1], target.shape[0])
    r_interpolated_translation_map = cv.resize(
        r_translation_map, target_shape, interpolation=cv.INTER_CUBIC
    )
    c_interpolated_translation_map = cv.resize(
        c_translation_map, target_shape, interpolation=cv.INTER_CUBIC
    )
    origin_img = stitch_block(obj_stack, target)
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
    nCMM_full_map = stitch_block(
        nCMM_stack, np.zeros([num_row * nr, num_col * nc]))
    if flag:
        nvec = int(np.min(target.shape) / 30)
        step = max(target.shape[0] // nvec, target.shape[1] // nvec)
        y, x = np.mgrid[: target.shape[0]: step, : target.shape[1]: step]
        r_ = r_interpolated_translation_map[::step, ::step]
        c_ = c_interpolated_translation_map[::step, ::step]
        plt.figure(figsize=(10, 10))
        plt.imshow(warp_img)
        plt.quiver(x, y, r_, c_, color="r", units="dots",
                   angles="xy", scale_units="xy")
        plt.title("Vector Field")
        plt.axis("off")
        plt.show()
        plt.figure(figsize=(10, 10))
        plt.imshow(nCMM_full_map, vmin=0, vmax=1)
        plt.title("nCMM")
        plt.axis("off")
        plt.show()
    if flag:
        compare_unreg_reg(origin_img, target, warp_img)
    rmax_shift = np.max(np.abs(r_interpolated_translation_map))
    cmax_shift = np.max(np.abs(c_interpolated_translation_map))
    return warp_img, rmax_shift, cmax_shift


# %%
new = tf.imread("new.tif", 0)
target = tf.imread("target.tif", 0)
# new = preprocess(new)
# target = preprocess(target)
# %%

# TODO: SIFT
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(new, None)
kp2, des2 = sift.detectAndCompute(target, None)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
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
ref_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
    -1, 1, 2
)
# 场景特征点坐标
sensed_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
    -1, 1, 2
)
# 方法采用RANSAC计算投影矩阵，阈值设为5.0，即误差的2范数超过5.0，视为局外点
H, status = cv.findHomography(
    ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
first_warped_image = cv.warpPerspective(
    new, H, (target.shape[1], target.shape[0]), borderMode=cv.BORDER_REPLICATE
)

# %%
block_row = 1
block_col = 1
new_block_stack = create_image_block_stack(
    first_warped_image, block_row, block_col)
target_block_stack = create_image_block_stack(target, block_row, block_col)
display_blocks(new_block_stack)
display_blocks(target_block_stack)
# %%
warp_img, rmax_shift, cmax_shift = blocks_registration(
    new_block_stack, target_block_stack, target, True
)
# %%
for i in range(5):
    for j in range(5):
        warp_block_stack = create_image_block_stack(
            warp_img, block_row, block_col)
        warp_img, rmax_shift, cmax_shift = blocks_registration(
            warp_block_stack, target_block_stack, target, True
        )
        print(rmax_shift, cmax_shift)
    block_row = block_row + 1
    block_col = block_col + 1
    target_block_stack = create_image_block_stack(target, block_row, block_col)
# %%
compare_unreg_reg(new, target, warp_img, True)
# %%
tf.imwrite("good.tif", transform(warp_img))
# %%
print(rmax_shift, cmax_shift)
# %%
