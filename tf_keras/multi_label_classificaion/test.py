import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import cv2
from tqdm import tqdm

x_train = []
x_test = []
y_train = []

df_train = pd.read_csv(r'D:\deep_learning_mehran\data\planet\train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread(r'D:\deep_learning_mehran\data\planet\train-jpg\{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append(targets)

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)

split = 35000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=4,
          verbose=1,
          validation_data=(x_valid, y_valid))

from sklearn.metrics import fbeta_score

p_valid = model.predict(x_valid, batch_size=128)
print(y_valid)
print(p_valid)
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))













images = []
classes = []
df_train = pd.read_csv(f'{data_dir}/{multi_label}')
folder_name = [i for i in os.listdir(data_dir) if os.path.isdir('/'.join([data_dir, i]))]
if len(folder_name) != 1:
    raise ValueError("only the image folder should exist in the data folder. Plz check it out! ")
for f, tags in df_train.values:

    img = f'{data_dir}/{folder_name[0]}/{f}.jpg'
    images.append(img)
    classes.append(tags.encode())

parsed_dataset = tf.data.Dataset.from_tensor_slices((images, classes))

def process(image, mask, n_classes=n_classes, label_to_class=label_to_class):

    load_image = tf.keras.preprocessing.image.load_img(image)
    im_np = tf.keras.preprocessing.image.img_to_array(load_image)

    targets = np.zeros(n_classes)
    for t in mask.split(' '):
        targets[label_to_class[t]] = 1

    return im_np, targets


def process(image, mask, n_classes=n_classes, label_to_class=label_to_class):
    image = tf.io.decode_jpeg(image, channels=3)

    return image, mask

dataset = parsed_dataset.map(process)

converted_dataset = parsed_dataset.map(lambda image,label:
                                       tf.py_function(func=process,
                                                      inp=[image,label],
                                                      Tout=[tf.string,tf.string]))

dataset = parsed_dataset.map(lambda image, label: process(image, label, num_classes=n_classes, label_to_class=label_to_class))

image_batch, label_batch = next(iter(converted_dataset))











list_ds = tf.data.Dataset.list_files((f'{data_dir}/{folder_name[0]}/*.jpg'), shuffle=False)
image_batch = next(iter(list_ds))

def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  file_name = tf.strings.split(parts[-1], '.')
  label = df_train[df_train['image_name'] == str(file_name[0].eval.decode())]['tags']
  return label

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_ds = list_ds.map(get_label)
image_batch = next(iter(train_ds))


df_train[df_train['image_name'] == 'train_0'

test = tf.data.Dataset.from_tensor_slices(dict(df_train))
image_batch = next(iter(test))




















