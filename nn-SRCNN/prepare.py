# %%
import glob
import h5py
import numpy as np
import tifffile as tf
print('OK!')
# %%


def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.hr_images_dir))):
        hr = tf.imread(image_path)
        hr = np.array(hr).astype(np.float32)
        for i in range(0, hr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, hr.shape[1] - args.patch_size + 1, args.stride):
                hr_patches.append(
                    hr[i:i + args.patch_size, j:j + args.patch_size])

    for image_path in sorted(glob.glob('{}/*'.format(args.lr_images_dir))):
        lr = tf.imread(image_path)
        lr = np.array(lr).astype(np.float32)
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(
                    lr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.hr_images_dir)))):
        hr = tf.imread(image_path)
        hr = np.array(hr).astype(np.float32)
        hr_group.create_dataset(str(i), data=hr)
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.lr_images_dir)))):
        lr = tf.imread(image_path)
        lr = np.array(lr).astype(np.float32)
        lr_group.create_dataset(str(i), data=lr)
    h5_file.close()
# %%


class Para_train(object):
    lr_images_dir = '../Data/10x_train'
    hr_images_dir = '../Data/20x_train'
    output_path = 'train.h5'
    patch_size = 50
    stride = 10


class Para_eval(object):
    lr_images_dir = '../Data/10x_eval'
    hr_images_dir = '../Data/10x_eval'
    output_path = 'eval.h5'


# %%
if __name__ == '__main__':
    args_train = Para_train()
    args_eval = Para_eval()
    train(args_train)
    eval(args_eval)

# %%
