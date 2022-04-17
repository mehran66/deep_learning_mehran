import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import albumentations as A
from pathlib import Path
from functools import partial
import time

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

    tensor_records_inputs = data_params['tensor_records']
    batch_size = data_params['batch_size'] * strg()[0].num_replicas_in_sync

    img_size = data_params['img_size']
    img_height = img_size[0]
    img_width = img_size[1]
    img_channel = img_size[2]

    augmentation = data_params['augmentation']

    classification_type = data_params['classification_type']

    if classification_type == 'multilabel':
        multi_label_csv = tfrecords_param['multi_label_csv']

    label_to_class = tfrecords_param['label_to_class']
    # create lables if the label_to_class input is None
    if label_to_class == None:
        raise ValueError("label_to_class cannot be None!")

    n_classes = tfrecords_param['n_classes']

    train_split=data_params['train_split']
    val_split=data_params['val_split']
    test_split=data_params['test_split']

    shuffle_size = data_params['shuffle_size']

    print("\nRead the inputs from the config.py file ...")
    print(f"data directory is: {data_dir}")
    print(f"plot directory is: {plot_dir}")
    print(f"tensor records flag is: {tensor_records_inputs}")
    print(f"batch size is: {batch_size}")
    print(f"image size is: {img_size}")
    print(f"image augmentation is: {augmentation}")
    print(f"multi_label is: {classification_type}")
    print(f"labels are is: {label_to_class}")
    print(f"number of classes are: {n_classes}")
    print(f"train split is: {train_split}")
    print(f"validation split is: {val_split}")
    print(f"test split is: {test_split}")
    print(f"shuffle size is: {shuffle_size}")

    # ===============================================================================
    # create the dataset
    class_to_label = {v: k for k, v in label_to_class.items()}
    assert n_classes == len(label_to_class)

    if tensor_records_inputs:
        print("\n******************************************************")
        print("Reading input data from tfrecords")
        # find the number of raw images encoded in the tf files
        tf_files = glob.glob(data_dir + r'/' + "*.tfrecords", recursive=False)
        nbr_images = sum(1 for _ in tf.data.TFRecordDataset(tf_files))
        print(f"there are {nbr_images} images encoded in the {len(tf_files)} input rfrecords")

        if classification_type != 'multilabel':
            parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode='classification')
        else:
            parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode='multilabel')

            df_train = pd.read_csv(f'{data_dir}/{multi_label_csv}')
            for label in label_to_class.keys():
                df_train[label] = df_train[df_train.columns[1]].apply(lambda x: 1 if label in x.split(' ') else 0)

    else:
        print("\n******************************************************")
        print("****Reading input data from the raw jpg images")
        if classification_type != 'multilabel':
            path = Path(data_dir)
            nbr_images = (sum(1 for x in path.glob('**/*.jpg') if x.is_file()))

            load_split = partial(
                tf.keras.preprocessing.image_dataset_from_directory,
                data_dir,
                validation_split=None,
                shuffle=True,
                seed=123,
                image_size=(img_height, img_width),
                batch_size=nbr_images,
                class_names=list(label_to_class.keys())
            )

            parsed_dataset = load_split()
            # ds_valid = load_split(subset='validation')

            assert parsed_dataset.class_names==list(label_to_class.keys())
            class_names = parsed_dataset.class_names
        else:

            df_train = pd.read_csv(f'{data_dir}/{multi_label_csv}')
            nbr_images = df_train.shape[0] - 1
            folder_name = [i for i in os.listdir(data_dir) if os.path.isdir('/'.join([data_dir, i]))]
            if len(folder_name) != 1:
                raise ValueError("only the image folder should exist in the data folder. Plz check it out! ")

            for label in label_to_class.keys():
                df_train[label] = df_train[df_train.columns[1]].apply(lambda x: 1 if label in x.split(' ') else 0)

            df_train['image_dir'] = df_train[df_train.columns[0]].apply(lambda x: os.path.join(data_dir, folder_name[0], x + '.jpg'))

            path_to_images = df_train['image_dir'].copy().values
            labels_array = df_train[label_to_class.keys()].copy().values
            parsed_dataset = tf.data.Dataset.from_tensor_slices((path_to_images, labels_array))

            def process_path(image, label):
                image = tf.io.read_file(image)
                image = tf.io.decode_jpeg(image, channels=3)

                return image, label

            parsed_dataset = parsed_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # image_batch, label_batch = next(iter(parsed_dataset))
    # image_batch
    # image = tf.io.read_file(image_batch)
    # image = tf.io.decode_raw(image, tf.uint8)
    #
    # image = tf.io.decode_jpeg(image, channels=3)
    # tf.io.parse_tensor(image, out_type=tf.uint8)
    #
    # image = tf.reshape(image, shape=[height, width, channel_image])
    # tf.io.decode_image(
    #     image,
    #     channels=None,
    #     dtype=tf.dtypes.uint8)
    #
    # tf.cast(image_batch, tf.uint8)
    # image = tf.io.read_file(tf.cast(image_batch, tf.uint8))



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

    # keep a copy of train_ds for assessment
    train_ds_t = train_ds

    print("\n******************************************************")
    print("train, validation and test data were generated")
    print("the number of images in the training dataset is:" + str(sum(1 for _ in train_ds)))
    print("the number of images in the validation dataset is:" + str(sum(1 for _ in val_ds)))
    print("the number of images in the test dataset is:" + str(sum(1 for _ in test_ds)))

    # ===============================================================================
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


        def aug_fn(image):
            data = {"image":image}
            aug_data = transforms(**data)
            aug_img = aug_data["image"]
            aug_img = tf.cast(aug_img, tf.float32)
            return aug_img


        @tf.function
        def process_data(image, label):
            aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
            return aug_img, label

        train_ds = train_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)

    #.shuffle(nbr_images) has eliminated as the data get shuffled before creating the dataset
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    print("\n******************************************************")
    print("data loaded successfully and datasets were generated")
    print(f"it took {(time.time() - start_time)} seconds to import data and create datasets")


    if assess_data:

        print("\n******************************************************")
        print("Assess imported data ...")

        # inspect training dataset

        # retrieve a single batch of 32 images.
        image_batch, label_batch = next(iter(train_ds))

        # image_batch = tensor of the shape (32, 64, 64, 3), image shape = 64 x 64 x 3 (height x width x channels)
        # label_batch = tensor of the shape (32,), (integer labels)
        print(f"image dataset shape in one batch is {image_batch.shape}")
        print(f"label dataset shape in one batch is {label_batch.shape}")
        print(f"image data type is {image_batch.dtype}")
        print(f"label data type is {label_batch.dtype}")

        # check image value range
        first_image = image_batch[0].numpy().astype("float32")
        print("range of values for a sample image is:")
        print(np.min(first_image), np.max(first_image))

        # visualize 10 images from dataset
        plt.figure(figsize=(20, 10))
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            image = (image_batch[i]/255).numpy().astype("float32") #float32
            plt.imshow(image)
            if classification_type != 'multilabel':
                plt.title(class_to_label[tf.round(label_batch[i]).numpy()])
            else:
                mykeys = np.where(label_batch[i].numpy() == 1)
                values = [class_to_label[x] for x in mykeys[0]]
                label = ', '.join(values)
                plt.title(label)
            plt.axis("off")
        #plt.show()
        plt.savefig(f'{plot_dir}/data_load_step_sample_images_lables.png')
        plt.close()

        if augmentation:
            # plot the impact of image augmentation
            image_batch, label_batch = next(iter(train_ds_t))
            tmp = 16 # nbr of images to plot
            if batch_size<tmp:
                tmp = batch_size
            plt.figure(figsize=(20, 10))
            for n in range(tmp):
                images, labels = process_data(image_batch, label_batch)
                ax = plt.subplot(4, 4, n + 1)
                image = (images / 255).numpy().astype("float32")  # float32
                plt.imshow(image)
                if classification_type != 'multilabel':
                    plt.title(class_to_label[tf.round(labels).numpy()])
                else:
                    mykeys = np.where(labels.numpy() == 1)
                    values = [class_to_label[x] for x in mykeys[0]]
                    label = ', '.join(values)
                    plt.title(label)
                plt.axis("off")
            plt.savefig(f'{plot_dir}/load_data_step_image_augmentation.png')
            plt.close()
            print("\nSome sample augmented images plotted in the following file. please check!")
            print(f'{plot_dir}/load_data_step_image_augmentation.png')

        if classification_type != 'multilabel':
            train_lables = []
            for i, l in train_ds:
                train_lables.extend(list(l.numpy()))

            val_lables = []
            for i, l in val_ds:
                val_lables.extend(list(l.numpy()))

            test_lables = []
            for i, l in test_ds:
                test_lables.extend(list(l.numpy()))

            _, train_count = np.unique(train_lables, return_counts=True)
            _, val_count = np.unique(val_lables, return_counts=True)
            _, test_count = np.unique(test_lables, return_counts=True)

            df = pd.DataFrame(data = (train_count,val_count,test_count))
            df = df.T
            df['Index']=list(label_to_class.keys())
            df.columns = ['Train', 'Validation','Test','Name']
            df

            plt.figure(figsize=(10, 10))
            plt.pie(train_count,
                   explode=(0,) * n_classes,
                   labels = list(label_to_class.keys()),
                   autopct = '%1.1f%%')
            plt.axis('equal')
            plt.title('Proportion of each observed quantity in train dataset')
            #plt.show()
            plt.savefig(f'{plot_dir}/data_load_step_label_distribution.png')
            plt.close()
            print("\nSome sample images and labels plotted in the following files. please check!")
            print(f'{plot_dir}/data_load_step_sample_images_lables.png')
            print(f'{plot_dir}/data_load_step_label_distribution.png')

        if classification_type == 'multilabel':
            labels = df_train[df_train.columns[1]].apply(lambda x: x.split(' '))
            from collections import Counter, defaultdict
            counts = defaultdict(int)
            for l in labels:
                for l2 in l:
                    counts[l2] += 1

            counts_df = pd.DataFrame.from_dict(counts, orient='index')
            counts_df.columns = ['count']
            counts_df.sort_values('count', ascending=False, inplace=True)

            fig, ax = plt.subplots()
            ax = ax.bar(counts_df.index, counts_df['count'])
            fig.set_size_inches(12, 8)
            plt.xticks(rotation=45)
            plt.savefig(f'{plot_dir}/data_load_step_label_distribution.png')
            plt.close()
            print("\ndistribution of labels were ploted in the following file!")
            print(f'{plot_dir}/data_load_step_label_distribution.png')

            def show_cooccurence_matrix(labels):
                numeric_df = df_train[labels]
                co_matrix = numeric_df.T.dot(numeric_df)

                fig, ax = plt.subplots(figsize=(7, 7))
                im = ax.imshow(co_matrix)
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
                ax.set_yticklabels(labels)
                cbar = fig.colorbar(im, ax=ax)  # not sure why this isn't working...
                #plt.show(fig)
                plt.savefig(f'{plot_dir}/data_load_step_cooccurence_matrix.png')
                plt.close()
                print("\ncooccurence matrix of labels were plotted in the following file!")
                print(f'{plot_dir}/data_load_step_cooccurence_matrix.png')

            # compute the co-ocurrence matrix
            show_cooccurence_matrix(list(label_to_class.keys()))

    print("\ndata loaded successfully")

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    load_data(assess_data=True)
