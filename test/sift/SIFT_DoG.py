# %%
import matplotlib.pyplot as plt
import numpy as np

print("OK!")
# %%


def convolve(filter, mat, padding, strides):

    filter_r, filter_c = filter.shape

    if len(mat.shape) == 3:
        mat_r, mat_c, mat_d = mat.shape
        channel = []
        for i in range(mat_d):
            pad_mat = np.pad(
                mat[:, :, i],
                ((padding[0], padding[1]), (padding[2], padding[3])),
                "constant",
            )
            temp = []
            for j in range(0, mat_r, strides[1]):
                temp.append([])
                for k in range(0, mat_c, strides[0]):
                    val = (filter * pad_mat[j: j +
                           filter_r, k: k + filter_c]).sum()
                    temp[-1].append(val)
            channel.append(np.array(temp))
        channel = tuple(channel)
        result = np.dstack(channel)
    else:
        mat_r, mat_c = mat.shape
        channel = []
        pad_mat = np.pad(
            mat, ((padding[0], padding[1]),
                  (padding[2], padding[3])), "constant"
        )
        for j in range(0, mat_r, strides[1]):
            channel.append([])
            for k in range(0, mat_c, strides[0]):
                val = (filter * pad_mat[j: j +
                       filter_r, k: k + filter_c]).sum()
                channel[-1].append(val)
    result = np.array(channel)
    return result


def GuassianKernel(sigma, dim):
    """
    :param sigma: Standard deviation
    :param dim: dimension(must be positive and also an odd number)
    :return: return the required Gaussian kernel.
    """
    temp = [t - (dim // 2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    result = (1.0 / (temp * np.pi)) * np.exp(
        -(assistant ** 2 + (assistant.T) ** 2) / temp
    )
    return result


def downsample(img, step=2):
    return img[::step, ::step]


# %%
def getDoG(img, n, sigma0, O=None):
    S = n + 3
    if O == None:
        O = int(np.log2(min(img.shape[0], img.shape[1]))) - 3

    k = 2 ** (1.0 / n)
    sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S)]
             for o in range(O)]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    GuassianPyramid = []
    for i in range(O):
        GuassianPyramid.append([])
        for j in range(S):
            dim = int(6 * sigma[i][j] + 1)  # 高斯核尺寸遵循3sigma原则
            if dim % 2 == 0:
                dim += 1
            GuassianPyramid[-1].append(
                convolve(
                    GuassianKernel(sigma[i][j], dim),
                    samplePyramid[i],
                    [dim // 2, dim // 2, dim // 2, dim // 2],
                    [1, 1],
                )
            )
    DoG = [
        [GuassianPyramid[o][s + 1] - GuassianPyramid[o][s]
            for s in range(S - 1)]
        for o in range(O)
    ]
    return DoG, GuassianPyramid  # %%


# %%
A = plt.imread("./SIFTimg/1-1.jpg")
# %%
DoG, GuassianPyramid = getDoG(A, 3, 1.5)
# %%


def zxy2xyz(img):
    Z = img.shape[0]
    X = img.shape[1]
    Y = img.shape[2]
    img_tran = np.zeros([X, Y, Z])
    for i in range(Z):
        img_tran[:, :, i] = img[i, :, :]
    return img_tran


# %%
A = zxy2xyz(DoG[1][1]).astype(np.uint8)
plt.figure(figsize=(int(np.log2(A.shape[0])), int(np.log2(A.shape[1]))))
plt.imshow(A)
plt.axis("off")
plt.show()
# %%
