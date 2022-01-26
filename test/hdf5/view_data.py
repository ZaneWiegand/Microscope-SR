# %%
import h5py
import matplotlib.pyplot as plt
print('OK!')
# %%
pic_train = h5py.File(
    '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/nn-SRCNN/train.h5', 'r')
# %%
print([key for key in pic_train.keys()], "\n")
# %%
plt.imshow(pic_train['hr'][202])
# %%
plt.imshow(pic_train['lr'][202])
# %%
pic_eval = h5py.File(
    '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/nn-SRCNN/eval.h5', 'r')
# %%
print([key for key in pic_eval.keys()], "\n")
# %%
print([key for key in pic_eval['hr'].keys()], "\n")
print([key for key in pic_eval['lr'].keys()], "\n")
# %%
plt.imshow(pic_eval['hr']['20'][:])
# %%
plt.imshow(pic_eval['lr']['20'][:])
# %%
