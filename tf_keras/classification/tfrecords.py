import sys

from functools import partial
import matplotlib.pyplot as plt
import tqdm
import glob
import os
from sklearn.utils import shuffle
import time


import tensorflow as tf
print(f'Tensorflow version is: {tf.__version__}')

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
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
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
    feature = tf.io.decode_jpeg(raw_image, channels=3) # tf.io.parse_tensor(raw_image, out_type=tf.uint8) and tf.io.decode_raw(sample['image'], tf.float64)
    if normalize:
       feature = tf.cast(feature, tf.float32) / 255.0
    feature = tf.reshape(feature, shape=[height, width, channel_image])
    print(feature)

    if mode == 'segmentation':
        label = tf.io.decode_png(label, channels=1)  # tf.io.parse_tensor(raw_image, out_type=tf.uint8)
        label = tf.reshape(label, shape=[height, width, channel_label])

    return (feature, label)



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

    start_time = time.time()

    # read the inputs from the config.py file
    mode = 'classification'

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']

    shard_size = tfrecords_param['shard_size']
    img_height = tfrecords_param['img_size'][0]
    img_width = tfrecords_param['img_size'][1]
    label_to_class = tfrecords_param['label_to_class']
    n_classes = tfrecords_param['n_classes']

    # define a arbitrary batch_size to test the outputs
    batch_size = 16

    class_to_label = {v: k for k, v in label_to_class.items()}
    assert n_classes == len(label_to_class)

    images = []
    classes = []
    for label_name in os.listdir(data_dir):
        if os.path.isdir('/'.join([data_dir, label_name])):
            cls = label_to_class[label_name]

            for img_name in os.listdir('/'.join([data_dir, label_name])):
                if img_name.endswith('jpg'):

                    images.append('/'.join([data_dir, label_name, img_name]))
                    classes.append(cls)

    images, classes = shuffle(images, classes, random_state=0)

    print(f"there are {len(images)} images in {n_classes} folders/classes")
    if len(images) == 0:
        print("There is no jpg images in the data folder")
        sys.exit()
    print("classes are:")
    print(label_to_class)

    assert len(images) == len(classes)

    print("*****images and labels were successfully imported****")

    write_images_to_tfr(images, classes, img_width, img_height, filename="training_data", max_files=shard_size, out_dir=data_dir, mode=mode)

    print("*****tfrecords were successfully generated****")

    print(f"it took {(time.time() - start_time)} seconds to generate tfrecords")

    if assess_data==True:

        parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode='classification')
        dataset = parsed_dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)



        def show_batch(image_batch, label_batch, name):
            plt.figure(figsize=(10, 10))
            for n in range(16):
                plt.subplot(4, 4, n + 1)
                plt.imshow(image_batch[n])
                plt.title(class_to_label[label_batch[n]])
                plt.axis("off")
            plt.savefig(f'{plot_dir}/{name}')

        image_batch, label_batch = next(iter(dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(), 'tf_records_step_sample_images_1.png')

        image_batch, label_batch = next(iter(dataset))
        show_batch(image_batch.numpy(), label_batch.numpy(), 'tf_records_step_sample_images_2.png')

        print("*****tfrecords were imported and tested. Some sample images and labels plotted in the following file. please check!****\n")
        print(f'{plot_dir}/tf_records_sample_images.png')

if __name__ == "__main__":
    tfrecord(assess_data=True)



    



