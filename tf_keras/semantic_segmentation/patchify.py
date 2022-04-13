import time
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

import tensorflow as tf
print(f'Tensorflow version is: {tf.__version__}')

from config import dir_params, preprocess_parames, patch_params

def process(images_dir, img, img_color_mode, img_size, padding, images_patched_dir, output_format = 'jpg'):
    load_image = tf.keras.preprocessing.image.load_img(f'{images_dir}/{img}', color_mode=img_color_mode)
    im_np = np.array(load_image)
    if len(im_np.shape) == 2:
        im_np = np.expand_dims(im_np, -1)
    im_np = np.expand_dims(im_np, 0)

    patches = tf.image.extract_patches(im_np,
                                       sizes=[1, img_size[0], img_size[1], 1],
                                       strides=[1, img_size[0], img_size[1], 1],
                                       rates=[1, 1, 1, 1],
                                       padding=padding)  # or VALID

    count_img = 0
    for imgs in patches:
        for r in range(imgs.shape[0]):
            for c in range(imgs.shape[1]):
                array = tf.reshape(imgs[r, c], shape=img_size).numpy().astype('uint8')
                if img_color_mode == 'grayscale':
                    im = Image.fromarray(array[...,-1])
                else:
                    im = Image.fromarray(array)
                im.save(f'{images_patched_dir}/{img[:-4]}_{r}_{c}.{output_format}')
                count_img += 1

    return count_img


def patchify(assess_data=True):

    print("\n####################################################################")
    print("#####################PATCHIFY IMAGES AND MASKS######################")
    print("######################################################################")

    start_time = time.time()

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']
    # raw images and masks
    images_dir = os.path.join(data_dir, preprocess_parames['preprocessed_images_dir_name'])
    masks_dir = os.path.join(data_dir, preprocess_parames['preprocessed_masks_dir_name'])

    # patched images and masks
    images_patched_dir = os.path.join(data_dir , patch_params['images_dir_name_patched'])
    masks_patched_dir =  os.path.join(data_dir , patch_params['masks_dir_name_patched'])
    if not os.path.isdir(images_patched_dir):
        os.mkdir(images_patched_dir)
    if not os.path.isdir(masks_patched_dir):
        os.mkdir(masks_patched_dir)

    # other inputs
    padding = patch_params['padding']
    img_size = patch_params['img_size']
    mask_size = patch_params['mask_size']

    img_nbr_channels = img_size[-1]
    if img_nbr_channels == 1:
        img_color_mode = 'grayscale'
    elif img_nbr_channels == 3:
        img_color_mode = 'rgb'
    elif img_nbr_channels == 4:
        img_color_mode = 'rgba'
    else:
        raise ValueError("number of image bands are not supported. it should be 1, 3 or 4")

    mask_nbr_channels = mask_size[-1]
    if mask_nbr_channels == 1:
        mask_color_mode = 'grayscale'
    elif mask_nbr_channels == 3:
        mask_color_mode = 'rgb'
    elif mask_nbr_channels == 4:
        mask_color_mode = 'rgba'
    else:
        raise ValueError("number of image bands are not supported. it should be 1, 3 or 4")

    print("\n###########################################")
    print("###importing data from config file...")
    print(f"plot directory: {plot_dir}")
    print(f"input image directory: {images_dir}")
    print(f"input mask directory: {masks_dir}")
    print(f"output patched image directory: {images_patched_dir}")
    print(f"output patched mask directory: {masks_patched_dir}")
    print(f"padding: {padding}")
    print(f"image size: {img_size}")
    print(f"mask: {mask_size}")
    print(f"image color mode: {img_color_mode}")
    print(f"maks color mode: {mask_color_mode}")


    # list of jpg images
    my_images = []
    dir_files_images = os.listdir(images_dir)
    dir_files_images = sorted(dir_files_images)

    for files in dir_files_images:
        if '.jpg' in files:
            my_images.append(files)

    # list of png masks
    my_masks = []
    dir_files_masks= os.listdir(masks_dir)
    dir_files_masks = sorted(dir_files_masks)

    for files in dir_files_masks:
        if '.png' in files:
            my_masks.append(files)

    assert len(my_images) == len(my_masks)

    print("\n###########################################")
    print(f"{len(my_images)} images and masks were imported")

    # patch images
    count_img = 0
    for img in tqdm(my_images, desc = 'patchify images ...'):
        cnt = process(images_dir, img, img_color_mode, img_size, padding, images_patched_dir, output_format = 'jpg')
        count_img += cnt


    # patch masks
    count_mask = 0
    for mask in tqdm(my_masks, desc = 'patchify masks ...'):
        cnt = process(masks_dir, mask, mask_color_mode, mask_size, padding, masks_patched_dir, output_format = 'png')
        count_mask += cnt

    assert count_img == count_mask

    print(f"{len(my_images)} large images were patched into {count_img} images")
    print(f"it took {(time.time() - start_time)} seconds")


    if assess_data == True:
        print("\n###########################################")
        print("assesses the generated patched data")

        # visualize patched images
        shutil.copy(f'{images_dir}/{my_images[0]}', f'{plot_dir}/patchify_step_sample_large_image.png')
        files = glob.glob(f'{images_patched_dir}/{my_images[0][:-4]}*.jpg', recursive=False)
        files_names = [os.path.basename(x) for x in files]
        files_names.sort()
        nbr_rows = int(files_names[-1][:-4].split('_')[-2]) + 1
        nbr_cols = int(files_names[-1][:-4].split('_')[-1]) + 1
        assert nbr_rows*nbr_cols == len(files_names)

        fig, axes = plt.subplots(nrows=nbr_rows, ncols=nbr_cols, figsize=(15, 15))
        img_count = 0
        for i in range(nbr_rows):
            for j in range(nbr_cols):
                load_image = tf.keras.preprocessing.image.load_img(files[img_count], color_mode=img_color_mode)
                im_np = np.array(load_image)
                axes[i, j].imshow(im_np)
                axes[i, j].axis('off')
                img_count += 1

        plt.savefig(f'{plot_dir}/patchify_step_sample_image.png')
        plt.close()
        print("\na sample patched image plotted in the following files. please check!")
        print(f'{plot_dir}/patchify_step_sample_patched_image.png')

        # visualize patched masks
        shutil.copy(f'{masks_dir}/{my_masks[0]}', f'{plot_dir}/patchify_step_sample_large_masks.png')
        files = glob.glob(f'{masks_patched_dir}/{my_masks[0][:-4]}*.png', recursive=False)
        files_names = [os.path.basename(x) for x in files]
        files_names.sort()
        nbr_rows = int(files_names[-1][:-4].split('_')[-2]) + 1
        nbr_cols = int(files_names[-1][:-4].split('_')[-1]) + 1
        assert nbr_rows*nbr_cols == len(files_names)

        fig, axes = plt.subplots(nrows=nbr_rows, ncols=nbr_cols, figsize=(15, 15))
        img_count = 0
        for i in range(nbr_rows):
            for j in range(nbr_cols):
                load_image = tf.keras.preprocessing.image.load_img(files[img_count], color_mode=img_color_mode)
                im_np = np.array(load_image)
                axes[i, j].imshow((im_np - im_np.min())/(im_np.max() - im_np.min()))
                axes[i, j].axis('off')
                img_count += 1

        plt.savefig(f'{plot_dir}/patchify_step_sample_mask.png')
        plt.close()
        print("a sample patched mask plotted in the following files. please check!")
        print(f'{plot_dir}/patchify_step_sample_patched_mask.png')

if __name__ == "__main__":
    patchify(assess_data=True)