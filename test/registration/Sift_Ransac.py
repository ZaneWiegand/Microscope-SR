# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tifffile as tf
print("OK!")
# %%


def transform(img):
    img = img / img.max() * 255
    img = img.astype(np.uint8)
    return img


# %%
new = tf.imread("new.tif", 0)
target = tf.imread("target.tif", 0)
# target_blur = cv.GaussianBlur(target, (13, 13), 0)
# cv.imwrite("target_blur.jpg", transform(target_blur))
# %%
sift = cv.SIFT_create()
# %%
kp1, des1 = sift.detectAndCompute(new, None)
kp2, des2 = sift.detectAndCompute(target, None)
# %%
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# %%
good_matches = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good_matches.append([m])
# %%
img = cv.drawMatchesKnn(
    new,
    kp1,
    target,
    kp2,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
# %%
plt.imshow(img)
# %%
cv.imwrite("matches.jpg", img)
# %%
good_matches = np.squeeze(good_matches)
# 物体特征点坐标
ref_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
    -1, 1, 2
)
# 场景特征点坐标
sensed_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
    -1, 1, 2
)
# %%
# 方法采用RANSAC计算投影矩阵，阈值设为5.0，即误差的2范数超过5.0，视为局外点
H, status = cv.findHomography(
    ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
# %%
warped_image = cv.warpPerspective(
    new, H, (target.shape[1], target.shape[0]), borderMode=cv.BORDER_REPLICATE
)
# %%
# cv.imwrite("warp_SiftRansac.jpg", warped_image)
# %%
