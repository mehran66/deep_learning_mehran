import tensorflow as tf
from pathlib import Path
import os
import numpy as np
import pandas as pd
import time

from config import dir_params, tfrecords_param, data_params, predict_param

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

    pred_dir = dir_params['pred_dir']
    model_check_points = dir_params['ckpt_dir']
    img_size = data_params['img_size']

    classification_type = data_params['classification_type']

    label_to_class = tfrecords_param['label_to_class']
    class_to_label = {v: k for k, v in label_to_class.items()}

    batch_size = predict_param['batch_size']

    model_name = predict_param['model_name']

    print("\nRead the inputs from the config.py file ...")
    print(f"prediction directory is: {pred_dir}")
    print(f"model check point directory is: {model_check_points}")
    print(f"classification type is: {classification_type}")
    print(f"image size is: {img_size}")
    print(f"label to class is: {label_to_class}")
    print(f"batch size is: {batch_size}")
    print(f"model name is: {model_name}")


    path = Path(pred_dir)
    nbr_images = (sum(1 for x in path.glob('**/*.jpg') if x.is_file()))
    print(f"\nthere are {nbr_images} jpg images in the prediction directory")

    # list all of the images
    my_images = []
    dir_files_images = os.listdir(pred_dir)
    dir_files_images = sorted(dir_files_images)

    for files in dir_files_images:
        if '.jpg' in files:
            my_images.append(files)

    print(f"\nload the model from: {model_check_points}/{model_name}")
    model = tf.keras.models.load_model(model_check_points + f'/{model_name}', compile=False)

    start_time = time.time()

    # create a generator
    a = [i for i in my_images]
    def gen():
        for i in range(len(a)):
            yield process(a[i], pred_dir, img_size)

    aDataset = tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(None,*img_size), dtype=tf.float32))
    pred = model.predict(aDataset,batch_size=batch_size)

    print(f"it took {(time.time() - start_time)} seconds to run prediction on {nbr_images} images")

    if classification_type == 'multiclasse':
        prob = [i.max() for i in pred]
        pred_class = [np.argmax(i) for i in pred]
        pred_label = [class_to_label[i] for i in pred_class]

    elif classification_type == 'binary':
        prob = [probas.tolist()[0] for probas in pred]
        pred_class = [int(np.round(probas).tolist()[0]) for probas in pred]
        pred_label = [class_to_label[i] for i in pred_class]

    elif classification_type == 'multilabel':
        pred_class = [np.where((probas > 0.5) == 1) for probas in pred]
        prob = [np.take(probas, cls).tolist()[0] for probas, cls in zip(pred, pred_class)]
        pred_label = [np.take(np.array(list(class_to_label.values())), i) for i in pred_class]
        pred_label = [', '.join(i.tolist()[0]) for i in pred_label]

    df = pd.DataFrame(list(zip(my_images, prob, pred_class, pred_label)),
                      columns=['images', 'prob', 'pred_class', 'pred_label'])

    if classification_type == 'multilabel':
        for label in label_to_class.keys():
            df[label] = pred[:,label_to_class[label]]

    print(f"\nsave the results to: {pred_dir}/predictions.csv")
    df.to_csv(f'{pred_dir}/predictions.csv')

    print(f"here is a few sample predictions:")
    print(df.head(10))

if __name__ == "__main__":
    predict()
