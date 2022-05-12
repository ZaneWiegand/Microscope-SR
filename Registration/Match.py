# %%
import cv2
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("OK!")
# %%
test_number = 10
pic1 = tf.imread(
    "./Raw-Data/10X/region{}.tif".format(
        test_number
    )
)
pic2 = tf.imread(
    "./Raw-Data/20X/region{}.tif".format(
        test_number
    )
)
# %%
pic1 = pic1[:, :, 1]
pic2 = pic2[:, :, 1]
# 插值pic1，变为原来的4倍
pic1_a, pic1_b = pic1.shape
pic1_new = cv2.resize(pic1, [pic1_b * 2, pic1_a * 2],
                      interpolation=cv2.INTER_LINEAR)
# %%
plt.imshow(pic1, cmap="gray")
# %%
plt.imshow(pic2, cmap="gray")
# %%
plt.imshow(pic1_new, cmap="gray")
# %%
# methods=[cv2.TM_SQDIFF_NORMED,cv2.TM_CCORR_NORMED,cv2.TM_CCOEFF_NORMED]
method = cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(pic1_new, pic2, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# %%
t1 = min_loc
pic2_a, pic2_b = pic2.shape
br = (t1[0] + pic2_a, t1[1] + pic2_b)
# %%
fig, ax = plt.subplots(1, 1)
ax.imshow(pic1_new, cmap="gray")
pic = fig.gca()
rect = patches.Rectangle(
    t1, pic2_b, pic2_a, linewidth=1, edgecolor="r", facecolor="none"
)
pic.add_patch(rect)
plt.show()
# %%
pic2_new = pic1_new[min_loc[0]: min_loc[0] +
                    pic2_a, min_loc[1]: min_loc[1] + pic2_b]
# %%
plt.imshow(pic2_new, cmap="gray")
# %%
tf.imwrite('10x10.tif', pic2_new)
tf.imwrite('20x10.tif', pic2)
# %%
