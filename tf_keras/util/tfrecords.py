import pandas as pd
import tensorflow as tf
from PIL import Image
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import tqdm
import glob
import os
from pathlib import Path
from sklearn.utils import shuffle



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
def write_images_to_tfr(images, labels, filename: str = "training_data", max_files: int = 10,
                             out_dir: str = "/content", mode = 'classification'):
    # determine the number of shards (single TFRecord files) we need:
    splits = (len(images) // max_files) + 1  # determine how many tfr shards are needed
    if len(images) % max_files == 0:
        splits -= 1
    print(f"\nUsing {splits} shard(s) for {len(images)} files, with up to {max_files} samples per shard")

    file_count = 0

    for i in tqdm.tqdm(range(splits)):
        current_shard_name = f"{out_dir}/{filename}_{i + 1}_{splits}.tfrecords"
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files:  # as long as our shard is not full
            # get the index of the file that we want to parse now
            index = i * max_files + current_shard_count
            if index == len(images):  # when we have consumed the whole data, preempt generation
                break

            current_image = images[index]
            current_label = labels[index]

            # create the required Example representation
            out = parse_single_image(image=current_image, label=current_label, mode = mode)

            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_count += 1

        writer.close()
    print(f"\nWrote {file_count} elements to TFRecord")
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

if __name__ == '__main__':
    #############################
    #### Segmentation (Binary) example
    ##############################

    dir = r"C:\Users\mehra\OneDrive\Desktop\deep_learning\data\building_footprint"
    image_dir = dir + r"\images_512"
    label_dir = dir + r"\masks_512"
    out_dir= dir

    mode = 'segmentation'
    batch_size = 8
    shard_size = 100

    image_names = glob.glob(f"{image_dir}/*.jpg")
    image_names.sort()
    images = np.array([np.array(Image.open(img)) for img in image_names])

    mask_names = glob.glob(f"{label_dir}/*.png")
    mask_names.sort()
    masks = np.array([np.expand_dims(np.array(Image.open(img).convert('L')), axis = 2) for img in mask_names])

    # create tfrecords files
    write_images_to_tfr(images, masks, filename = "training_data",  max_files=shard_size, out_dir = out_dir , mode = 'segmentation')

    # Read tfrecords files and create a dataset
    parsed_dataset = get_dataset(tfr_dir=out_dir, pattern="*.tfrecords", mode='segmentation')
    dataset = parsed_dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    # visualize images and labels
    image_batch, label_batch = next(iter(dataset))
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(8):
            ax = plt.subplot(4, 2, (n*2)+1)
            plt.imshow(image_batch[n])
            ax = plt.subplot(4, 2, (n*2)+2)
            plt.imshow(label_batch[n])
            plt.axis("off")
    show_batch(image_batch.numpy(), label_batch.numpy())


    #############################
    #### multi class classification example
    ##############################
    data_dir= r'C:\Users\mehra\OneDrive\Desktop\deep_learning\data\EuroSAT'
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    batch_size = 8
    shard_size = 10000

    '''
    path = Path(data_dir)
    file_count = (sum(1 for x in path.glob('**/*') if x.is_file()))
    
    
    load_split = partial(
        tf.keras.preprocessing.image_dataset_from_directory,
        data_dir,
        validation_split=None,
        shuffle=True,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=file_count,
    )
    
    ds_train = load_split()
    #ds_valid = load_split(subset='validation')
    
    class_names = ds_train.class_names
    print("\nClass names: {}".format(class_names))
    
    for image_batch, labels_batch in ds_train:
      print(image_batch.shape)
      print(labels_batch.shape)
      break
    
    write_images_to_tfr(image_batch.numpy(), labels_batch.numpy(), filename = "training_data",  max_files=shard_size, out_dir = data_dir , mode = 'classification')
    
    parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode='classification')
    dataset = parsed_dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    
    
    image_batch, label_batch = next(iter(dataset))
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(8):
            ax = plt.subplot(4, 2, n+1)
            plt.imshow(image_batch[n])
            plt.title(ds_train.class_names[label_batch[n]])
            plt.axis("off")
    
    show_batch(image_batch.numpy(), label_batch.numpy())
    '''

    label_to_class = {
        'AnnualCrop': 0,
        'Forest': 1,
        'HerbaceousVegetation': 2,
        'Highway': 3,
        'Industrial': 4,
        'Pasture': 5,
        'PermanentCrop': 6,
        'Residential': 7,
        'River': 8,
        'SeaLake': 9
    }
    class_to_label = {v: k for k, v in label_to_class.items()}
    n_classes = len(label_to_class)


    Images = []
    Classes = []

    for label_name in os.listdir(data_dir):
        cls = label_to_class[label_name]

        for img_name in os.listdir('/'.join([data_dir, label_name])):
            if img_name.endswith('jpg'):
                img = tf.keras.utils.load_img('/'.join([data_dir, label_name, img_name]), target_size=(IMG_WIDTH, IMG_HEIGHT))
                img = tf.keras.utils.img_to_array(img)

                Images.append(img)
                Classes.append(cls)

    Images = np.array(Images, dtype=np.uint8)
    Classes = np.array(Classes, dtype=np.uint8)
    Images, Classes = shuffle(Images, Classes, random_state=0)
    Images.shape, Classes.shape

    write_images_to_tfr(Images, Classes, filename = "training_data",  max_files=shard_size, out_dir = data_dir , mode = 'classification')

    parsed_dataset = get_dataset(tfr_dir=data_dir, pattern="*.tfrecords", mode='classification')
    dataset = parsed_dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)


    image_batch, label_batch = next(iter(dataset))
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(8):
            ax = plt.subplot(4, 2, n+1)
            plt.imshow(image_batch[n])
            plt.title(class_to_label[label_batch[n]])
            plt.axis("off")

    show_batch(image_batch.numpy(), label_batch.numpy())

    #############################
    #### Segmentation (multi class - RGB labels - large tiles) example
    ##############################
    batch_size = 8

    dir = r'C:\Users\mehra\OneDrive\Desktop\deep_learning\data\semantic_drone_dataset_semantics_v1.1\semantic_drone_dataset\training_set'

    images = dir + r"\images"
    masks = dir + r'\gt\semantic\label_images'
    class_dict = dir + r'gt\semantic\class_dict.csv'

    images_512 = dir + r'images_512'
    masks_512 =  dir + r'masks_512'

    out_dir = dir

    my_images = []
    dir_files_images = os.listdir(images)
    dir_files_images = sorted(dir_files_images)

    for files in dir_files_images:
        if '.jpg' in files:
            my_images.append(files)

    my_masks = []
    dir_files_masks= os.listdir(masks)
    dir_files_masks = sorted(dir_files_masks)

    for files in dir_files_masks:
        if '.png' in files:
            my_masks.append(files)

    assert len(my_images) == len(my_masks)

    # patch images
    count = 0
    for img in my_images:
        load_image = tf.keras.preprocessing.image.load_img(f'{images}/{img}')
        im_np = np.array(load_image)
        im_np = np.expand_dims(im_np, 0)

        patches = tf.image.extract_patches(im_np,
                                          sizes = [1, 512, 512, 1],
                                          strides = [1, 512, 512, 1],
                                          rates=[1, 1, 1, 1],
                                          padding = 'SAME') # or VALID

        for imgs in patches:
            for r in range(imgs.shape[0]):
                for c in range(imgs.shape[1]):
                    array = tf.reshape(imgs[r,c], shape = (512,512,3)).numpy().astype('uint8')
                    count+=1
                    im = Image.fromarray(array)
                    im.save(f'{images_512}/{my_images[0][:-4]}_{r}_{c}.jpg')

    # patch masks
    count = 0
    for img in my_masks:
        load_image = tf.keras.preprocessing.image.load_img(f'{masks}/{img}')
        im_np = np.array(load_image)
        im_np = np.expand_dims(im_np, 0)

        patches = tf.image.extract_patches(im_np,
                                          sizes = [1, 512, 512, 1],
                                          strides = [1, 512, 512, 1],
                                          rates=[1, 1, 1, 1],
                                          padding = 'SAME') # or VALID

        for imgs in patches:
            for r in range(imgs.shape[0]):
                for c in range(imgs.shape[1]):
                    array = tf.reshape(imgs[r,c], shape = (512,512,3)).numpy().astype('uint8')
                    count+=1
                    im = Image.fromarray(array)
                    im.save(f'{masks_512}/{my_masks[0][:-4]}_{r}_{c}.png')



    mask_names = glob.glob(f"{masks_512}/*.png")
    mask_names.sort()
    masks = np.array([np.array(Image.open(img)) for img in mask_names])

    annotations = pd.read_csv(class_dict)
    annotations= annotations.sort_values('name',ignore_index=True)

    for i in range(len(masks)):
        label = masks[i]
        for row in annotations.iterrows():
            label[(label == (row[1][1], row[1][2], row[1][3])).all(axis=2)] = row[0]
        masks[i] = label

    masks = [np.expand_dims(mask[...,-1], axis = 2) for mask in masks]

    image_names = glob.glob(f"{images_512}/*.jpg")
    image_names.sort()
    images = np.array([np.array(Image.open(img)) for img in image_names])




    write_images_to_tfr(images, masks, filename = "training_data",  max_files=10000, out_dir = out_dir , mode = 'segmentation')

    parsed_dataset = get_dataset(tfr_dir=out_dir, pattern="*.tfrecords", mode='segmentation')
    dataset = parsed_dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)


    image_batch, label_batch = next(iter(dataset))
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(8):
            ax = plt.subplot(4, 2, (n*2)+1)
            plt.imshow(image_batch[n])
            ax = plt.subplot(4, 2, (n*2)+2)
            plt.imshow(label_batch[n])
            plt.axis("off")

    show_batch(image_batch.numpy(), label_batch.numpy())



