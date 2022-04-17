import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import albumentations as A
import time
import collections
import json
import tensorflow as tf

from config import dir_params, tfrecords_param, data_params
from tfrecords import get_dataset

def strg():

    # Detect hardware, return appropriate distribution strategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
        print('\nRunning on TPU ', tpu.master())
        print("Num TPUs Available: ", len(tf.config.experimental.list_physical_devices('TPU')))
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()  # default distribution strategy in Tensorflow. Works on CPU and single GPU.

    print("\nnumber of replicas: ", strategy.num_replicas_in_sync)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

    return strategy, tpu

def load_data(assess_data=True):

    print("\n####################################################################")
    print("#####################LOAD DATA#####################################")
    print("######################################################################")

    start_time = time.time()

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']

    img_size = data_params['img_size']
    mask_size = data_params['mask_size']

    scale = data_params['scale']

    mask_to_categorical_flag = data_params['mask_to_categorical']

    tensor_records_inputs = data_params['tensor_records_dir']
    if tensor_records_inputs == None:
        images_dir = os.path.join(data_dir , data_params['ml_images_dir_name'])
        masks_dir =  os.path.join(data_dir , data_params['ml_masks_dir_name'])
        print(f"image directory is: {images_dir}")
        print(f"mask directory is: {masks_dir}")

    batch_size = data_params['batch_size'] * strg()[0].num_replicas_in_sync

    augmentation = data_params['augmentation']

    label_to_class_file = os.path.join(data_dir , data_params['label_to_class'])
    with open(label_to_class_file, 'r') as myfile:
        label_to_class = json.load(myfile)
    n_classes = data_params['n_classes']

    train_split=data_params['train_split']
    val_split=data_params['val_split']
    test_split=data_params['test_split']
    shuffle_size=data_params['shuffle_size']

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

    print("\nRead the inputs from the config.py file ...")
    print(f"data directory is: {data_dir}")
    print(f"plot directory is: {plot_dir}")
    print(f"tfrecords flag is: {tensor_records_inputs}")
    print(f"batch size is: {batch_size}")
    print(f"image size is: {img_size}")
    print(f"mask size is: {mask_size}")
    print(f"image color mode is: {img_color_mode}")
    print(f"mask color mode is: {mask_color_mode}")
    print(f"scale data is: {scale}")
    print(f"mask_to_categorical mode is: {mask_to_categorical_flag}")
    print(f"image augmentation is: {augmentation}")
    print(f"labels are: {label_to_class}")
    print(f"number of classes are: {n_classes}")
    print(f"train split is: {train_split}")
    print(f"validation split is: {val_split}")
    print(f"test split is: {test_split}")
    print(f"shuffle size is: {shuffle_size}")

    # ===============================================================================
    # create the dataset

    if tensor_records_inputs:
        print("\n******************************************************")
        print("Reading input data from tfrecords")
        # find the number of raw images encoded in the tf files
        tf_files = glob(data_dir + r'/' + "*.tfrecords", recursive=False)
        nbr_images = sum(1 for _ in tf.data.TFRecordDataset(tf_files))
        print(f"there are {nbr_images} images encoded in the {len(tf_files)} input rfrecords")
        parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode='segmentation', img_color_mode = img_color_mode, mask_color_mode = mask_color_mode)

    else:
        print("\n******************************************************")
        print("****Reading input data from the raw jpg images and png masks")
        images = sorted(glob(images_dir + "/*.jpg"))
        masks = sorted(glob(masks_dir + "/*.png"))
        nbr_images = len(images)
        nbr_masks = len(masks)
        assert nbr_images == nbr_masks

        print(f"There are {nbr_images} images imported")

        @tf.function
        def load_data(image, label):

            raw_image = tf.io.read_file(image)
            image = tf.io.decode_jpeg(raw_image, channels=3)

            raw_label = tf.io.read_file(label)
            label = tf.io.decode_png(raw_label, channels=1)

            return image, label

        parsed_dataset = tf.data.Dataset.from_tensor_slices((images, masks))
        parsed_dataset = parsed_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    @tf.function
    def cast_resize_data(image, label):

        image.set_shape([None, None, 3])
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(images=image, size=[img_size[0], img_size[1]])

        label.set_shape([None, None, 1])
        label = tf.cast(label, tf.float32)
        label = tf.image.resize(images=label, size=[mask_size[0], mask_size[1]], method='nearest')

        return image, label

    @tf.function
    def scale_data(image, label):
        image = tf.math.divide(image, 255.0)
        return image, label

    @tf.function
    def mask_to_categorical(image, label, num_classes=7):
        label = tf.one_hot(tf.cast(tf.squeeze(label), tf.int32), num_classes)
        label = tf.cast(label, tf.float32)
        label.set_shape([None, None, num_classes])
        return image, label

    '''
    # this code convert the labels into an acceptable format. I figured out that running functions on tf datasets is vry time consuming and it is best to prepare data in advance
    
    @tf.function
    def normalize_lable(image, label):
        return image, tf.cast(label/255, tf.uint8)

    @tf.function
    def label_rgb2unique(image, label):
        cnt = 0
        for i in label_to_class:
            label = tf.where(
                tf.expand_dims(tf.reduce_all(tf.equal(label, label_to_class[i]), axis=-1), axis=2),
                tf.constant(cnt, tf.int32),
                tf.cast(label, tf.int32))
            cnt+=1
        label = tf.expand_dims(tf.cast(label, tf.uint8)[...,-1], axis=2)
        return image,label


    # change (0,255) labels to (0,1)
    if mask_size[2] == 1:
        lbl = list(label_to_class.values())
        if n_classes == 2 and (255 in label_to_class.values()):
            parsed_dataset = parsed_dataset.map(normalize_lable, num_parallel_calls=tf.data.AUTOTUNE)
            for i in label_to_class:
                if label_to_class[i] == 255:
                    label_to_class[i] = 1

    #Change RGB labels to a one band mask with labels of 0,1,2,...
    elif mask_size[2] == 3:
        parsed_dataset = parsed_dataset.map(label_rgb2unique, num_parallel_calls=tf.data.AUTOTUNE)
        cnt = 0
        for i in label_to_class:
            label_to_class[i] = cnt
            cnt+=1

    with open('label_to_class.json', 'w') as f:
        json.dump(label_to_class, f, indent=2)

    '''

    class_to_label = {v: k for k, v in label_to_class.items()}


    # generate training/validation/test dataset
    assert (train_split + test_split + val_split) == 1

    if shuffle_size != 0:
        # Specify seed to always have the same split distribution between runs
        parsed_dataset = parsed_dataset.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * nbr_images)
    val_size = int(val_split * nbr_images)

    train_ds = parsed_dataset.take(train_size)
    val_ds = parsed_dataset.skip(train_size).take(val_size)
    test_ds = parsed_dataset.skip(train_size).skip(val_size)

    # a temporary dataset be used for some plots
    train_ds_t = parsed_dataset

    print("\n******************************************************")
    print("train, validation and test data were generated")
    print("the number of images in the training dataset is: " + str(sum(1 for _ in train_ds)))
    print("the number of images in the validation dataset is: " + str(sum(1 for _ in val_ds)))
    print("the number of images in the test dataset is: " + str(sum(1 for _ in test_ds)))

    # ===============================================================================
    # data augmentation
    # https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/20/Pixel-level-transforms-using-albumentations-package.html
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/

    # data augmentation
    # https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/20/Pixel-level-transforms-using-albumentations-package.html
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/

    if augmentation:

        transforms = A.Compose([

            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()], p=0.5),

            A.RandomRotate90(p=0.5),

            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       always_apply=False, p=0.2),

            A.Blur (blur_limit=5, always_apply=False, p=0.2),

            A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.2),

            A.ChannelShuffle(p=0.1),

            A.FancyPCA(alpha=1, p=0.1),

            A.GaussNoise(var_limit=(10.0, 50.0), mean=-50, p=0.1),

            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.1),

            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.1),

            A.HueSaturationValue(50, 50, 20,  p=0.1),

            A.CoarseDropout(p=0.1)
        ], p=.5)


        def aug_fn(image, label):
            data = {"image": image, "mask": label}
            aug_data = transforms(**data)
            aug_img = aug_data["image"]
            aug_mask = aug_data['mask']
            return aug_img, aug_mask

        def process_data(image, label):
            aug_img, aug_mask = tf.numpy_function(func=aug_fn, inp=[image, label], Tout=[tf.uint8,tf.uint8])
            return aug_img, aug_mask

        train_ds = train_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(cast_resize_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(cast_resize_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(cast_resize_data, num_parallel_calls=tf.data.AUTOTUNE)

    if scale:
        train_ds = train_ds.map(scale_data, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(scale_data, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(scale_data, num_parallel_calls=tf.data.AUTOTUNE)

    if mask_to_categorical_flag:
        train_ds = train_ds.map(lambda image, label: mask_to_categorical(image, label, num_classes=n_classes))
        val_ds = val_ds.map(lambda image, label: mask_to_categorical(image, label, num_classes=n_classes))
        test_ds = test_ds.map(lambda image, label: mask_to_categorical(image, label, num_classes=n_classes))


    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    print("\n******************************************************")
    print("data loaded successfully and datasets were generated")
    print(f"it took {(time.time() - start_time)} seconds to import data and create datasets")

    if assess_data:
        print("\n******************************************************")
        print("Assess imported data ....")

        # inspect training dataset

        # retrieve a single batch of 32 images.
        image_batch, label_batch = next(iter(train_ds))

        print(f"image dataset shape in one batch is {image_batch.shape}")
        print(f"label dataset shape in one batch is {label_batch.shape}")
        print(f"image data type is {image_batch.dtype}")
        print(f"label data type is {label_batch.dtype}")

        # check image value range
        first_image = image_batch[0].numpy().astype("float32")
        print("range of values for a sample image are:")
        print(np.min(first_image), np.max(first_image))

        # check mask value range
        first_image = label_batch[0].numpy().astype("float32")
        print("unique values for a sample label are:")
        print(np.unique(first_image))

        if ~scale:
            image_batch = image_batch/255.0

        if mask_to_categorical_flag:
            label_batch = tf.argmax(label_batch, axis=-1)

        # plot sample images from dataset
        def show_batch(image_batch, label_batch, name):
            tmp = 8 # nbr of images to plot
            if batch_size<8:
                tmp = batch_size
            plt.figure(figsize=(10, 10))
            for n in range(tmp):
                ax = plt.subplot(4, 4, (n*2)+1)
                ax.axis("off")
                plt.imshow(image_batch[n])
                ax = plt.subplot(4, 4, (n * 2) + 2)
                ax.axis("off")
                plt.imshow(np.squeeze(label_batch[n]))
            plt.savefig(f'{plot_dir}/{name}')
            plt.close()

        show_batch(image_batch.numpy(), label_batch.numpy(), 'load_data_step_sample_images.png')
        print("\nSome sample images and labels plotted in the following file. please check!")
        print(f'{plot_dir}/load_data_step_sample_images.png')

        if augmentation:
            # plot the impact of image augmentation
            image_batch, label_batch = next(iter(train_ds_t))
            tmp = 8 # nbr of images to plot
            if batch_size<8:
                tmp = batch_size
            plt.figure(figsize=(10, 10))
            for n in range(tmp):
                images, labels = process_data(image_batch, label_batch)
                ax = plt.subplot(4, 4, (n * 2) + 1)
                ax.axis("off")
                plt.imshow(images)
                ax = plt.subplot(4, 4, (n * 2) + 2)
                ax.axis("off")
                plt.imshow(np.squeeze(labels))
            plt.savefig(f'{plot_dir}/load_data_step_image_augmentation.png')
            plt.close()
            print("\nSome sample augmented images plotted in the following file. please check!")
            print(f'{plot_dir}/load_data_step_image_augmentation.png')

        # check and plot the distribution of data in each class
        train_lables = [collections.Counter(l.numpy().flatten()) for i, l in train_ds_t]
        train_count = collections.Counter()
        for i in range(len(train_lables)):
            train_count += train_lables[i]

        for i in class_to_label.keys():
            if not i in train_count.keys():
                train_count[i] = 0

        train_count = dict(sorted(train_count.items()))

        print(f'distribution of data in each class is: {train_count}')

        plt.bar(train_count.keys(), train_count.values())
        plt.xticks(list(train_count.keys()))
        # plt.show()
        plt.savefig(f'{plot_dir}/load_data_step_label_distribution_barchart.png')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.pie(train_count.values(), labels=list(label_to_class.keys()), autopct='%1.1f%%')
        plt.title('Proportion of each observed quantity in dataset')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{plot_dir}/load_data_step_label_distribution_pieplot.png')
        plt.close()

        print("\nSome sample images and labels plotted in the following files. please check!")
        print(f'{plot_dir}/load_data_step_label_distribution_barchart.png')
        print(f'{plot_dir}/load_data_step_label_distribution_pieplot.png')

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    load_data(assess_data=True)