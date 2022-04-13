import tensorflow as tf
from pathlib import Path
import os
import numpy as np
import time
import json
from PIL import Image
import random
from skimage import io
import pandas as ps

from config import dir_params, data_params, model_params, predict_param
from postprocess import find_threshold, measure_mask


def process(img, data_dir, img_size):
    load_image = tf.keras.preprocessing.image.load_img(f'{data_dir}/{img}', target_size=(img_size[0], img_size[1]))
    im_np = tf.keras.preprocessing.image.img_to_array(load_image)
    im_np = np.expand_dims(im_np, 0)
    return im_np

def predict():

    '''
    The following reference propose a solution for multiprocessing but it seems that the model.predict already used all of the cpus while processing a dataset

    Also, it seems that there is a known bug in model.predict that leads to memory leakage and the following solutions have been proposed:

    x_test = np.vstack([process(i, pred_dir, img_size) for i in my_images])
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    pred1 = model.predict(x_test)
    pred2 = model(x_test, training=False) #output is a tensor
    pred3 = model.predict_on_batch(x_test)
    gc.collect()

    '''

    print("\n####################################################################")
    print("#####################PREDICTION#####################################")
    print("######################################################################")

    pred_dir_input = dir_params['pred_dir_input']
    pred_dir_output = dir_params['pred_dir_output']
    data_dir = dir_params['data_dir']
    model_check_points = dir_params['ckpt_dir']
    img_size = data_params['img_size']
    multiclass = model_params['multiclass']

    label_to_class_file = os.path.join(data_dir, data_params['label_to_class'])
    with open(label_to_class_file, 'r') as myfile:
        label_to_class = json.load(myfile)
    n_classes = data_params['n_classes']
    class_to_label = {v: k for k, v in label_to_class.items()}

    batch_size = predict_param['batch_size']
    binary_threshold = predict_param['binary_threshold']

    model_name = predict_param['model_name']

    measure = predict_param['measure']

    print("\nRead the inputs from the config.py file ...")
    print(f"prediction input directory is: {pred_dir_input}")
    print(f"prediction output is: {pred_dir_output}")
    print(f"data directory is: {data_dir}")
    print(f"model check point directory is: {model_check_points}")
    print(f"image size is: {img_size}")
    print(f"multiclass flag is: {multiclass}")
    print(f"label to class is: {label_to_class}")
    print(f"number of class is: {n_classes}")
    print(f"batch size is: {batch_size}")
    print(f"binary threshold is: {binary_threshold}")
    print(f"model name is: {model_name}")
    print(f"measure is: {measure}")

    path = Path(pred_dir_input)
    nbr_images = (sum(1 for x in path.glob('**/*.jpg') if x.is_file()))
    print(f"there are {nbr_images} jpg images in the prediction directory")

    # list all of the images
    my_images = []
    dir_files_images = os.listdir(pred_dir_input)
    dir_files_images = sorted(dir_files_images)

    for files in dir_files_images:
        if files.endswith('.jpg'):
            my_images.append(files)

    print(f"\nload the model from: {model_check_points}/{model_name}")
    model = tf.keras.models.load_model(model_check_points + f'/{model_name}', compile=False)

    start_time = time.time()

    # create a generator
    def gen():
        for i in range(len(my_images)):
            yield process(my_images[i], pred_dir_input, img_size)

    dataset = tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(None,*img_size), dtype=tf.float32))
    pred = model.predict(dataset, batch_size=batch_size)

    print(f"it took {(time.time() - start_time)} seconds to run prediction on {nbr_images} images")

    if binary_threshold == 'calculate':
        nbr_sample = 10 if 10<nbr_images else nbr_images
        samples = random.sample(range(0, nbr_images), nbr_sample)
        lst_threshold = []
        for s in samples:
            lst_threshold.append(find_threshold(pred[s]))

        binary_threshold = np.array(lst_threshold).mean()


    if multiclass:
        y_pred_argmax = np.argmax(pred, axis=3)
    else:
        if binary_threshold is not None:
            y_pred_argmax = np.where(pred > binary_threshold, 255, 0)
        else:
            y_pred_argmax = pred


    if multiclass or binary_threshold is not None:
        for i, p in zip(my_images, y_pred_argmax):
            im = Image.fromarray(p.astype(np.uint))
            im.save(f'{pred_dir_output}/{i}'.replace('jpg','png'))
    else:
        for i, p in zip(my_images, y_pred_argmax):
            im = Image.fromarray(np.squeeze((np.round(p*255))).astype(np.uint))
            im.save(f'{pred_dir_output}/{i}'.replace('jpg','png'))

    if measure:
        for i, p in zip(my_images, y_pred_argmax):
            image = io.imread(f'{pred_dir_input}/{i}')
            if multiclass or binary_threshold is not None:
                _, label_image, df_measurments = measure_mask(image, np.squeeze(p), probability=False)
            else:
                _, label_image, df_measurments = measure_mask(image, np.squeeze(p), probability=True)

            im = Image.fromarray(label_image.astype(np.uint))
            im.save(f'{pred_dir_output}/{i[:-4]}_measure.png')
            df_measurments.to_json(f'{pred_dir_output}/{i[:-4]}_measure.json')

if __name__ == "__main__":

    predict()



'''
# Production

https://www.youtube.com/watch?v=HXzz87WVm6c&list=PLZsOBAyNTZwbIjGnolFydAN33gyyGP7lT&index=243&ab_channel=DigitalSreeni

https://medium.com/devseed/technical-walkthrough-packaging-ml-models-for-inference-with-tf-serving-2a50f73ce6f8

https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-tensorflow-models-in-aws-lambda/
'''