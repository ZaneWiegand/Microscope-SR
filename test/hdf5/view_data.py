# %%
import h5py
import matplotlib.pyplot as plt
print('OK!')
# %%
pic_train = h5py.File('train.h5', 'r')
# %%
print([key for key in pic_train.keys()], "\n")
# %%
plt.imshow(pic_train['hr'][100])
# %%
plt.imshow(pic_train['lr'][100])
# %%
pic_eval = h5py.File('eval.h5', 'r')
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
