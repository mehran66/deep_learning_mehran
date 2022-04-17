import sys
from functools import partial
import matplotlib.pyplot as plt
import tqdm
import glob
import os
import time
import numpy as np
import tensorflow as tf

from config import dir_params, tfrecords_param


# References
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# https://keras.io/examples/keras_recipes/creating_tfrecords/
# https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/tfrecord.py
# https://www.kaggle.com/ryanholbrook/tfrecords-basics

# assumptions:
# images are in jpg format and labels are in png format

#############################
####define features#######
##############################

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


#############################
####Create one example#######
##############################
def parse_single_image(image, label, mode='classification'):
    # define the dictionary -- the structure -- of our single example

    if mode == 'classification':
        data = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'channel_image': _int64_feature(image.shape[2]),
            'raw_image': _bytes_feature(tf.io.encode_jpeg(image)),
            # tf.io.serialize_tensor or tf.io.encode_jpeg or tf.io.encode_png or tf.io.encode_jpeg(image).numpy(), image.tobytes()
            'label': _int64_feature(label)
        }

    elif mode == 'segmentation':
        data = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'channel_image': _int64_feature(image.shape[2]),
            'channel_label': _int64_feature(label.shape[2]),
            'raw_image': _bytes_feature(tf.io.encode_jpeg(image)),
            # tf.io.serialize_tensor or tf.io.encode_jpeg or tf.io.encode_png
            'label': _bytes_feature(tf.io.encode_png(label))
        }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


#####################################
####TFRecord Writer with Shrad#######
#####################################
def write_images_to_tfr(images, labels, img_width, img_height, img_color_mode, mask_color_mode, filename: str = "training_data", max_files: int = 10,
                        out_dir: str = "/content", mode='classification'):
    # determine the number of shards (single TFRecord files) we need:
    splits = (len(images) // max_files) + 1  # determine how many tfr shards are needed
    if len(images) % max_files == 0:
        splits -= 1
    print("\n#########################################")
    print(f"Using {splits} shard(s) for {len(images)} files, with up to {max_files} samples per shard")

    file_count = 0

    for i in tqdm.tqdm(range(splits)):
        current_shard_name = f"{out_dir}/{filename}_{i + 1}_{splits}.tfrecords"

        if os.path.exists(current_shard_name):
            print("ERROR: tfrecords are already existing in the output directory. Please check!")
            sys.exit()

        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files:  # as long as our shard is not full
            # get the index of the file that we want to parse now
            index = i * max_files + current_shard_count
            if index == len(images):  # when we have consumed the whole data, preempt generation
                break

            img = tf.keras.utils.load_img(images[index], target_size=(img_width, img_height), color_mode=img_color_mode)
            current_image = tf.keras.utils.img_to_array(img, dtype='uint8')

            mask = tf.keras.utils.load_img(labels[index], target_size=(img_width, img_height), color_mode=mask_color_mode)
            current_label = tf.keras.utils.img_to_array(mask, dtype='uint8')
            if current_label.ndim == 2:
                current_label = np.expand_dims(current_label, axis=2)


            # create the required Example representation
            out = parse_single_image(image=current_image, label=current_label, mode=mode)

            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_count += 1

        writer.close()
    print(f"\nWrote {file_count} elements to {splits} TFRecord")
    print(f"tfrecords saved in {out_dir}")
    return file_count


#############################
#### Read TFRecord#######
##############################

def parse_tfr_element(element, normalize=False, mode='segmentation', img_color_mode='rgb', mask_color_mode='grayscale'):
    # use the same structure as above; it's kinda an outline of the structure we now want to create

    if mode == 'classification':
        data = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'raw_image': tf.io.FixedLenFeature([], tf.string),
            'channel_image': tf.io.FixedLenFeature([], tf.int64),
        }
    elif mode == 'segmentation':
        data = {
            'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'raw_image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'channel_image': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'channel_label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    channel_image = content['channel_image']
    label = content['label']
    raw_image = content['raw_image']

    if mode == 'segmentation':
        channel_label = content['channel_label']

    # get our 'feature'-- our image -- and reshape it appropriately
    if img_color_mode == 'grayscale':
        feature = tf.io.decode_jpeg(raw_image, channels=1)  # tf.io.parse_tensor(raw_image, out_type=tf.uint8) and tf.io.decode_raw(sample['image'], tf.float64)
    elif img_color_mode == 'rgb':
        feature = tf.io.decode_jpeg(raw_image, channels=3)
    elif img_color_mode == 'rgba':
        feature = tf.io.decode_jpeg(raw_image, channels=4)

    if normalize:
        feature = tf.cast(feature, tf.float32) / 255.0
    feature = tf.reshape(feature, shape=[height, width, channel_image])

    if mode == 'segmentation':
        if mask_color_mode == 'grayscale':
            label = tf.io.decode_png(label, channels=1)  # tf.io.parse_tensor(raw_image, out_type=tf.uint8)
        elif mask_color_mode == 'rgb':
            label = tf.io.decode_png(label, channels=3)
        elif mask_color_mode == 'rgba':
            label = tf.io.decode_png(label, channels=4)
        label = tf.reshape(label, shape=[height, width, channel_label])

    return (feature, label)


def get_dataset(tfr_dir: str = "/content", pattern: str = "*.tfrecords", mode='segmentation', img_color_mode='rgb', mask_color_mode='grayscale'):
    files = glob.glob(tfr_dir + r'/' + pattern, recursive=False)

    nbr_images = sum(1 for _ in tf.data.TFRecordDataset(files))
    print(f"there are {nbr_images} images encoded in the input rfrecords")

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        files, num_parallel_reads=tf.data.AUTOTUNE
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(parse_tfr_element, mode=mode, img_color_mode=img_color_mode, mask_color_mode=mask_color_mode), num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset


#############################
#### call the functions to create the TFRecords#######
##############################

def tfrecord(assess_data=True):

    print("\n######################################################")
    print("#####################Create TfRecords######################")
    print("######################################################")

    start_time = time.time()

    # read the inputs from the config.py file
    mode = 'segmentation'

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']

    images_dir = os.path.join(data_dir , tfrecords_param['ml_images_dir_name'])
    masks_dir =  os.path.join(data_dir , tfrecords_param['ml_masks_dir_name'])

    shard_size = tfrecords_param['shard_size']
    img_size = tfrecords_param['img_size']
    img_height = img_size[0]
    img_width = img_size[1]

    mask_size = tfrecords_param['mask_size']

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

    print("\n#########################################")
    print("importing data from config file...")
    print(f"plot directory: {plot_dir}")
    print(f"input image directory: {images_dir}")
    print(f"input mask directory: {masks_dir}")
    print(f"shrad size: {shard_size}")
    print(f"image size: {img_size}")
    print(f"mask: {mask_size}")
    print(f"image color mode: {img_color_mode}")
    print(f"maks color mode: {mask_color_mode}")

    # define an arbitrary batch_size to test the outputs
    batch_size = 16

    # list images
    image_names = glob.glob(f"{images_dir}/*.jpg")
    image_names.sort()

    # list masks
    mask_names = glob.glob(f"{masks_dir}/*.png")
    mask_names.sort()

    # create tfrecords files
    write_images_to_tfr(image_names, mask_names, img_width, img_height, img_color_mode, mask_color_mode, filename="training_data", max_files=shard_size, out_dir=data_dir,
                        mode=mode)
    print("\ntfrecords were successfully generated")
    print(f"it took {(time.time() - start_time)} seconds to generate tfrecords")

    if assess_data == True:

        parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode=mode, img_color_mode=img_color_mode, mask_color_mode=mask_color_mode)
        dataset = parsed_dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)

        def show_batch(image_batch, label_batch, name):
            plt.figure(figsize=(10, 10))
            for n in range(8):
                ax = plt.subplot(4, 4, (n*2)+1)
                plt.imshow(image_batch[n])
                ax = plt.subplot(4, 4, (n * 2) + 2)
                plt.imshow(np.squeeze(label_batch[n]))
                plt.axis("off")
            plt.savefig(f'{plot_dir}/{name}')
            plt.close()

        image_batch, label_batch = next(iter(dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(), 'tf_records_step_sample_images_1.png')

        image_batch, label_batch = next(iter(dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(), 'tf_records_step_sample_images_2.png')

        print("\n#########################################")
        print("tfrecords were imported and tested. Some sample images and labels plotted in the following file. please check!")
        print(f'{plot_dir}/tf_records_step_sample_images_1.png')
        print(f'{plot_dir}/tf_records_step_sample_images_2.png')


if __name__ == "__main__":
    tfrecord(assess_data=True)










