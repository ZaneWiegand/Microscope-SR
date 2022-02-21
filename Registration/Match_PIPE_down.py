# %%
import cv2 as cv
import tifffile as tf
import numpy as np
from Match_PIPE import preprocess, global_registration, create_image_block_stack, blocks_registration
import warnings
warnings.filterwarnings("ignore")  # 忽略警告
print("OK!")
# %%
if __name__ == "__main__":
    for pic_number in np.arange(15):
        test_number = pic_number+1
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

        # ? 合适阈值
        pre_threshold = 0
        pic10x = preprocess(pic10x, pre_threshold)
        pic20x = preprocess(pic20x, pre_threshold)

        # 所需pic20x
        pic20x_r, pic20x_c = pic20x.shape
        pic20x_ex = cv.resize(
            pic20x, [pic20x_c // 2, pic20x_r // 2], interpolation=cv.INTER_LINEAR
        )
        method = cv.TM_SQDIFF_NORMED
        result = cv.matchTemplate(pic10x, pic20x_ex, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        t1 = min_loc
        pic20x_ex_r, pic20x_ex_c = pic20x_ex.shape
        pic10x_cut = pic10x[
            min_loc[0]: min_loc[0] + pic20x_ex_r, min_loc[1]: min_loc[1] + pic20x_ex_c
        ]

        new = pic10x_cut
        target = pic20x_ex

        # * method = "SIFT","ORB","KAZE","AKAZE","BRISK","nCCM"
        global_method = "SIFT"
        warp_img = global_registration(new, target, global_method, True)
        first_warp_img = warp_img

        block_row_ini = block_row = 1
        block_col_ini = block_col = 1
        target_block_stack = create_image_block_stack(
            target, block_row, block_col)
        # display_blocks(target_block_stack)

        threshold = 0.25
        layer = 16
        layer_state = np.ones(layer)
        warp_img_layer = np.zeros(
            [warp_img.shape[0], warp_img.shape[1], layer])
        target_img_layer = np.zeros([target.shape[0], target.shape[1], layer])
        target = target.astype(np.float64)
        warp_img = warp_img.astype(np.float64)
        pic20x_ex = pic20x_ex.astype(np.float64)
        method = cv.INTER_LINEAR

        for i in range(layer):
            count = 0
            rtmap = np.zeros_like(target)
            ctmap = np.zeros_like(target)
            warp_img_layer[:, :, i] = warp_img  # i=0,是SIFT矫正的底层图
            target_img_layer[:, :, i] = target
            rmax_shift = cmax_shift = 1
            while ~(rmax_shift < threshold and cmax_shift < threshold):
                target = (target + warp_img + pic20x_ex) / 3
                warp_block_stack = create_image_block_stack(
                    warp_img, block_row, block_col)
                warp_img, rmax_shift, cmax_shift, rtmap, ctmap = blocks_registration(
                    warp_block_stack, target_block_stack, target, method, False
                )
                print("第{}层位移: (r = {}, c = {})".format(
                    i + 1, rmax_shift, cmax_shift))
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

            target_block_stack = create_image_block_stack(
                target, block_row, block_col)

        tup = [(i, layer_state[i]) for i in range(len(layer_state))]
        k = np.max([j for j, n in tup if n == 1])
        print("max layer: {}".format(k + 1))
        for j, n in tup:
            if n == 1:
                print(
                    "layer: {} , block size: {}".format(
                        j + 1,
                        (
                            int(np.ceil(
                                target.shape[0] / (j + block_row_ini))),
                            int(np.ceil(
                                target.shape[1] / (j + block_col_ini))),
                        ),
                    )
                )

        if method == cv.INTER_CUBIC:
            methods = "CUBIC"
        elif method == cv.INTER_LINEAR:
            methods = "LINEAR"
        elif method == cv.INTER_NEAREST:
            methods = "NEAREST"
        else:
            methods = "ERROR"
        tf.imwrite(
            "./Registration-down/test{}_pic_10x_{}_{}_{}_{}_{}divide.tif".format(
                test_number, pre_threshold, global_method, methods, threshold, k + 1
            ),
            warp_img.astype(np.uint8),
        )
        tf.imwrite(
            "./Registration-down/test{}_pic10x.tif".format(test_number),
            pic10x_cut.astype(np.uint8),
        )
        tf.imwrite(
            "./Registration-down/test{}_pic_20x.tif".format(
                test_number), pic20x.astype(np.uint8)
        )
