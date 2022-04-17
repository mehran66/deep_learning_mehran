import sys
from functools import partial
import matplotlib.pyplot as plt
import tqdm
import glob
import os
from sklearn.utils import shuffle
import time
import pandas as pd
import numpy as np
import tensorflow as tf

from config import dir_params, tfrecords_param, data_params

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
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_featur_list(value):
    """Returns an int64_list from a bool / enum / int / uint list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


#############################
####Create one example#######
##############################
def parse_single_image(image, label, mode = 'classification'):
    # define the dictionary -- the structure -- of our single example

    if mode == 'classification':
        data = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'channel_image': _int64_feature(image.shape[2]),
            'raw_image': _bytes_feature(tf.io.encode_jpeg(image)), # tf.io.serialize_tensor or tf.io.encode_jpeg or tf.io.encode_png or tf.io.encode_jpeg(image).numpy(), image.tobytes()
            'label': _int64_feature(label)
        }

    if mode == 'multilabel':
        data = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'channel_image': _int64_feature(image.shape[2]),
            'raw_image': _bytes_feature(tf.io.encode_jpeg(image)), # tf.io.serialize_tensor or tf.io.encode_jpeg or tf.io.encode_png or tf.io.encode_jpeg(image).numpy(), image.tobytes()
            #'label': _bytes_feature(str.encode(label)) # This is for a string case
            'label': _int64_featur_list(label.tolist())
        }

    elif mode == 'segmentation':
        data = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'channel_image': _int64_feature(image.shape[2]),
            'channel_label': _int64_feature(label.shape[2]),
            'raw_image': _bytes_feature(tf.io.encode_jpeg(image)), # tf.io.serialize_tensor or tf.io.encode_jpeg or tf.io.encode_png
            'label': _bytes_feature(tf.io.encode_png(label))
        }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

#####################################
####TFRecord Writer with Shrad#######
#####################################
def write_images_to_tfr(images, labels, img_width, img_height, filename: str = "training_data", max_files: int = 10,
                             out_dir: str = "/content", mode = 'classification'):

    # determine the number of shards (single TFRecord files) we need:
    splits = (len(images) // max_files) + 1  # determine how many tfr shards are needed
    if len(images) % max_files == 0:
        splits -= 1
    print(f"\nUsing {splits} shard(s) for {len(images)} files, with up to {max_files} samples per shard")

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

            img = tf.keras.utils.load_img(images[index], target_size=(img_width, img_height))
            current_image = tf.keras.utils.img_to_array(img)

            current_label = labels[index]

            # create the required Example representation
            out = parse_single_image(image=current_image, label=current_label, mode = mode)

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

def parse_tfr_element(element, normalize=False, mode='segmentation'):
    # use the same structure as above; it's kinda an outline of the structure we now want to create


    if mode == 'classification':
        data = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'raw_image': tf.io.FixedLenFeature([], tf.string),
            'channel_image': tf.io.FixedLenFeature([], tf.int64),
        }

    if mode == 'multilabel':
        data = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            #'label': tf.io.FixedLenFeature([], tf.string), #This is for a string case
            'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
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
    image = tf.io.decode_jpeg(raw_image, channels=3) # tf.io.parse_tensor(raw_image, out_type=tf.uint8) and tf.io.decode_raw(sample['image'], tf.float64)
    if normalize:
       image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, shape=[height, width, channel_image])

    if mode == 'segmentation':
        label = tf.io.decode_png(label, channels=1)  # tf.io.parse_tensor(raw_image, out_type=tf.uint8)
        label = tf.reshape(label, shape=[height, width, channel_label])

    return (image, label)


def get_dataset(tfr_dir: str = "/content", pattern: str = "*.tfrecords", mode='segmentation'):
    files = glob.glob(tfr_dir + r'/' +  pattern, recursive=False)

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
        partial(parse_tfr_element, mode=mode), num_parallel_calls=tf.data.AUTOTUNE
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

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']

    classification_type = data_params['classification_type']

    if classification_type == 'multilabel':
        multi_label_csv = tfrecords_param['multi_label_csv']

    mode = 'multilabel' if classification_type=='multilabel' else 'classification'

    shard_size = tfrecords_param['shard_size']
    img_height = tfrecords_param['img_size'][0]
    img_width = tfrecords_param['img_size'][1]
    label_to_class = tfrecords_param['label_to_class']

    n_classes = tfrecords_param['n_classes']
    class_to_label = {v: k for k, v in label_to_class.items()}
    assert n_classes == len(label_to_class)

    # define an arbitrary batch_size to test the outputs
    batch_size = 16

    print("\n#########################################")
    print("importing data from config file...")
    print(f"plot directory: {plot_dir}")
    print(f"data directory: {data_dir}")
    print(f"classification type is: {data_dir}")
    print(f"mode: {mode}")
    print(f"shrad size: {shard_size}")
    print(f"image height: {img_height}")
    print(f"image width: {img_width}")
    print(f"label_to_class: {label_to_class}")
    print(f"number of classes: {n_classes}")

    images = []
    classes = []
    if classification_type != 'multilabel':
        for label_name in os.listdir(data_dir):
            if os.path.isdir('/'.join([data_dir, label_name])):
                cls = label_to_class[label_name]

                for img_name in os.listdir('/'.join([data_dir, label_name])):
                    if img_name.endswith('jpg'):

                        images.append('/'.join([data_dir, label_name, img_name]))
                        classes.append(cls)
    else:
        df_train = pd.read_csv(f'{data_dir}/{multi_label_csv}')
        folder_name = [i for i in os.listdir(data_dir) if os.path.isdir('/'.join([data_dir, i]))]
        if len(folder_name) != 1:
            raise ValueError("only the image folder should exist in the data folder. Plz check it out! ")

        for label in label_to_class.keys():
            df_train[label] = df_train[df_train.columns[1]].apply(lambda x: 1 if label in x.split(' ') else 0)

        df_train['image_dir'] = df_train[df_train.columns[0]].apply(lambda x: os.path.join(data_dir, folder_name[0], x + '.jpg'))

        images = df_train['image_dir'].copy().values
        classes = df_train[label_to_class.keys()].copy().values

    images, classes = shuffle(images, classes, random_state=0)

    print("\nRead images and labels ...")
    print(f"there are {len(images)} images in {n_classes} folders/classes")
    if len(images) == 0:
        print("There is no jpg images in the data folder")
        sys.exit()

    assert len(images) == len(classes)

    write_images_to_tfr(images, classes, img_width, img_height, filename="training_data", max_files=shard_size, out_dir=data_dir, mode=mode)

    print("\ntfrecords were successfully generated")

    print(f"it took {(time.time() - start_time)} seconds to generate tfrecords")

    if assess_data==True:
        print("\n#########################################")
        print("\nAssessing tfrecords ...")

        parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode=mode)
        dataset = parsed_dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)

        def show_batch(image_batch, label_batch, name):
            plt.figure(figsize=(20, 10))
            for n in range(16):
                plt.subplot(4, 4, n + 1)
                plt.imshow(image_batch[n])
                if classification_type != 'multilabel':
                    plt.title(class_to_label[label_batch[n]])
                else:
                    # plt.title(label_batch[n].decode()) # this is for string label
                    mykeys = np.where(label_batch.numpy() == 1)
                    values = [class_to_label[x] for x in mykeys[0]]
                    label = ', '.join(values)
                    plt.title(label)
                plt.axis("off")
            plt.savefig(f'{plot_dir}/{name}')

        try:
            image_batch, label_batch = next(iter(dataset))
            show_batch(image_batch.numpy(), label_batch.numpy(), 'tf_records_step_sample_images_1.png')

            image_batch, label_batch = next(iter(dataset))
            show_batch(image_batch.numpy(), label_batch.numpy(), 'tf_records_step_sample_images_2.png')
        except:
            print(f"number of images are smaller than the requested dataset for plots")

        print("*****tfrecords were imported and tested. Some sample images and labels plotted in the following file. please check!****\n")
        print(f'{plot_dir}/tf_records_sample_images.png')

if __name__ == "__main__":
    tfrecord(assess_data=True)



