# %%
import h5py
import matplotlib.pyplot as plt
print('OK!')
# %%
pic_train = h5py.File('train.h5', 'r')
# %%
print([key for key in pic_train.keys()], "\n")
# %%
plt.imshow(pic_train['data'][1][0])
# %%
plt.imshow(pic_train['label'][1][0])
# %%
pic_eval = h5py.File(
    '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/nn-SRCNN/eval.h5', 'r')
# %%
print([key for key in pic_eval.keys()], "\n")
# %%
print([key for key in pic_eval['hr'].keys()], "\n")
print([key for key in pic_eval['lr'].keys()], "\n")
# %%
plt.imshow(pic_eval['hr']['40'][:])
# %%
plt.imshow(pic_eval['lr']['40'][:])
# %%
demo_train = h5py.File(
    '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/test/hdf5/91-image_x2.h5')
demo_eval = h5py.File(
    '/Users/zanewiegand/代码/python/Microscope-Super-Resolution/test/hdf5/Set5_x2.h5'
)
# %%
print([key for key in demo_train.keys()], "\n")
print([key for key in demo_eval.keys()], "\n")
# %%
print([key for key in demo_train.keys()], "\n")
print([key for key in demo_eval['hr'].keys()], "\n")
print([key for key in demo_eval['lr'].keys()], "\n")
# %%
plt.imshow(demo_train['hr'][22])
# %%
plt.imshow(demo_train['lr'][22])
# %%
plt.imshow(demo_eval['hr']['2'])
# %%
plt.imshow(demo_eval['lr']['2'])
# %%
