import time
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import shutil
import json
from tqdm import tqdm
import cv2

import tensorflow as tf
print(f'Tensorflow version is: {tf.__version__}')

from config import dir_params, preprocess_parames, patch_params



def generate_border(image_file, border_size=5, n_erosions=1, object_value=1):

    image = np.array(Image.open(image_file))

    erosion_kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(image, erosion_kernel, iterations=n_erosions)

    kernel_size = 2 * border_size + 1
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(eroded_image, dilation_kernel, iterations=1)
    dilated_image_2 = np.where(dilated_image == object_value, object_value+1, dilated_image)

    image_with_border = np.where(eroded_image >= object_value, object_value, dilated_image_2)

    return image_with_border

def preprocess(assess_data=True):

    print("\n####################################################################")
    print("#####################PREPROCESS IMAGES AND MASKS######################")
    print("######################################################################")

    start_time = time.time()

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']

    input_images = os.path.join(data_dir, preprocess_parames['input_images_dir_name'])
    input_masks = os.path.join(data_dir, preprocess_parames['input_masks_dir_name'])

    preprocessed_images = os.path.join(data_dir, preprocess_parames['preprocessed_images_dir_name'])
    preprocessed_masks = os.path.join(data_dir, preprocess_parames['preprocessed_masks_dir_name'])
    if not os.path.isdir(preprocessed_images):
        os.mkdir(preprocessed_images)
    if not os.path.isdir(preprocessed_masks):
        os.mkdir(preprocessed_masks)

    img_size = preprocess_parames['img_size']
    mask_size = preprocess_parames['mask_size']

    n_classes = preprocess_parames['n_classes']
    label_to_class = preprocess_parames['label_to_class']

    create_border = preprocess_parames['create_border']

    print("\n########################################")
    print("declare parameters")
    print(f"data directory is: {data_dir}")
    print(f"plot directory is: {plot_dir}")
    print(f"input images are in: {input_images}")
    print(f"input masks are in: {input_masks}")
    print(f"process images will save in: {preprocessed_images}")
    print(f"process masks will save in: {preprocessed_masks}")
    print(f"image size would be: {img_size}")
    print(f"mask size would be: {mask_size}")
    print(f"number of classes are: {n_classes}")
    print(f"labels are: {label_to_class}")
    print(f"make border is: {create_border}")

    assert img_size == img_size

    print("\n########################################")
    print("check and process the format and size of images ...")
    print("png is converted to jpg and image size is checked and resized if needed")
    cnt_img = 0
    if os.path.isdir(input_images):
        png_flag = False
        resize_flag = False
        for img_name in tqdm(os.listdir(input_images), desc = 'check images ...'):

            # check png files
            if img_name.endswith('png'):
                png_flag = True
                im1 = Image.open('/'.join([input_images, img_name]))
                if im1.mode in ("RGBA"):
                    im1 = im1.convert("RGB")
                if im1.size != img_size[0:2]:
                    im1 = im1.resize(img_size[0:2])
                    resize_flag = True
                cnt_img += 1
                im1.save('/'.join([preprocessed_images, img_name[:-4] + '.jpg']))

            # check jpg files
            if img_name.endswith('jpg'):
                im1 = Image.open('/'.join([input_images, img_name]))
                if im1.size != img_size[0:2]:
                    im1 = im1.resize(img_size[0:2])
                    resize_flag = True
                cnt_img += 1
                im1.save('/'.join([preprocessed_images, img_name[:-4] + '.jpg']))

    else:
        raise ValueError(f"{input_images} is not a directory")

    print(f"{cnt_img} images were processed and transferred to the {preprocessed_images} folder")
    if png_flag:
        print("png images were converted to jpg images")
    if resize_flag:
        print(f"images were resized to {img_size}")

    print("\n########################################")
    print("check and process the format and size of masks ...")
    print("jpg is converted to png and image size is checked and resized if needed")
    cnt_mask = 0
    if os.path.isdir(input_masks):
        jpg_flag = False
        resize_flag = False
        for img_name in tqdm(os.listdir(input_masks), desc = 'check masks ...'):

            # check png files
            if img_name.endswith('jpg'):
                im1 = Image.open('/'.join([input_masks, img_name]))
                jpg_flag = True
                if im1.size != img_size[0:2]:
                    im1 = im1.resize(img_size[0:2], resample=Image.NEAREST)
                    resize_flag = True
                im1.save('/'.join([preprocessed_masks, img_name[:-4] + '.png']))
                cnt_mask += 1

            # check jpg files
            if img_name.endswith('png'):
                im1 = Image.open('/'.join([input_masks, img_name]))
                if im1.size != mask_size[0:2]:
                    im1 = im1.resize(mask_size[0:2], resample=Image.NEAREST)
                    resize_flag = True
                im1.save('/'.join([preprocessed_masks, img_name[:-4] + '.png']))
                cnt_mask += 1

    else:
        raise ValueError(f"{input_masks} is not a directory")

    print(f"{cnt_mask} masks were processed and transferred to the {preprocessed_masks} folder")
    if jpg_flag:
        print("ipg masks were converted to png images")
    if resize_flag:
        print(f"masks were resized to {mask_size}")

    assert cnt_img == cnt_mask

    print("\n########################################")
    print("check masks and recreate them if needed. It is required to have one band with unique values for each class like 0,1,2,3,...")

    lbl = list(label_to_class.keys())
    cls = list(label_to_class.values())
    range_list = [i for i in range(len(lbl))]
    map_values = {range_list[i]: cls[i] for i in range(len(range_list))}
    map_lbls = {lbl[i]: range_list[i] for i in range(len(range_list))}

    if isinstance(cls[0], list):
        nbr_chl = 3
    else:
        nbr_chl = 1

    if mask_size[2] == 1 or mask_size[2] == 3:
        tmp = cls.copy()
        tmp.sort()
        if tmp == range_list:
            print("masks are in good shape and there is no need for any changes")
        else:
            for img_name in tqdm(os.listdir(preprocessed_masks), desc = 'recreate masks ...'):
                if img_name.endswith('png'):
                    im1 = np.array(Image.open('/'.join([preprocessed_masks, img_name])))
                    for row in map_values:
                        if nbr_chl == 1:
                            #np.where(im1 == map_values[row], row, im1)
                            im1[(im1 == map_values[row])] = row
                        else:
                            im1[(im1 == map_values[row]).all(axis=-1)] = row

                    if nbr_chl != 1:
                        im1 = im1[..., -1]
                    Image.fromarray(im1).save('/'.join([preprocessed_masks, img_name[:-4] + '.png']))

            print("masks were recreated with the following mapping function")
            print(map_values)

    else:
        raise ValueError("The code currently only supports masks with 1 or 3 bands")

    with open(os.path.join(data_dir , 'label_to_class.json'), 'w') as f:
        json.dump(map_lbls, f, indent=2)

    print(f"labels to class dictinary were exported as a json file in: {os.getcwd()}/label_to_class.json")


    if create_border:
        for img_name in tqdm(os.listdir(preprocessed_masks), desc='make borders ...'):
            if img_name.endswith('png'):
                processed_image = generate_border('/'.join([preprocessed_masks, img_name]), border_size=5, n_erosions=1)
                Image.fromarray(processed_image).save('/'.join([preprocessed_masks, img_name[:-4] + '.png']))

    print("\n########################################")
    print(f"\nit took {(time.time() - start_time)} seconds to preprocess data")

    if assess_data == True:
        # plot some images
        pass


if __name__ == "__main__":
    preprocess(assess_data=True)

