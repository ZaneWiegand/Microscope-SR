# %%
import cv2 as cv
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np
# %%
test_number = 10
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

# pic10x_r, pic10x_c = pic10x.shape
# pic10x_ex = cv.resize(
#     pic10x, [pic10x_c * 2, pic10x_r * 2], interpolation=cv.INTER_CUBIC
# )

pic10x_ex = pic10x
pic20x_r, pic20x_c = pic20x.shape
pic20x = cv.resize(
    pic20x, [pic20x_c // 2, pic20x_r // 2], interpolation=cv.INTER_CUBIC
)
# %%
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


def global_registration(obj, ref, method, flag=False):
    # TODO: SIFT
    if method == "SIFT":
        global_form = cv.SIFT_create()
    elif method == "ORB":
        global_form = cv.ORB_create()
    elif method == "KAZE":
        global_form = cv.KAZE_create()
    elif method == "AKAZE":
        global_form = cv.AKAZE_create()
    elif method == "BRISK":
        global_form = cv.BRISK_create()
    else:
        return "ERROR"
    kp1, des1 = global_form.detectAndCompute(obj, None)
    kp2, des2 = global_form.detectAndCompute(ref, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    if flag:
        img = cv.drawMatchesKnn(
            obj,
            kp1,
            ref,
            kp2,
            good_matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        img_obj = cv.drawKeypoints(
            obj, kp1, obj, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        img_ref = cv.drawKeypoints(
            ref, kp1, ref, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img_obj)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_ref)
        plt.axis("off")
        plt.tight_layout()
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    good_matches = np.squeeze(good_matches)
    # 物体特征点坐标
    ref_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    # 场景特征点坐标
    sensed_matched_kpts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    # 方法采用RANSAC计算投影矩阵，阈值设为5.0，即误差的2范数超过5.0，视为局外点
    H, status = cv.findHomography(
        ref_matched_kpts, sensed_matched_kpts, cv.RANSAC, 5.0)
    warp_img = cv.warpPerspective(
        obj, H, (ref.shape[1], ref.shape[0]
                 ), borderMode=cv.BORDER_REPLICATE
    )
    # TODO: SIFT end
    return warp_img


# %%
warp = global_registration(new, target, 'SIFT')
# %%
tf.imwrite('10x10.tif', warp)
# %%
tf.imwrite('20x10.tif', target)
# %%
