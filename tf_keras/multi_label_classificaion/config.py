'''
Assumptions
- Images are 3 band RGB in jpg format
- Images are stored in different folders corresponding to each class. Folder name is used as class name
- For multilabel classification, data folder includes the image folder with all images as 3 band RGB in jpg format and
    one csv file that maps image name (with extension) to labels (space delimited labels). The csv file should have only two
    columns (image_name and tag)
'''

import os
from os import path as op
import pandas as pd


# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Set data directory
data_dir = r'D:\deep_learning_mehran\data\planet_test'
# prediction directory where images are stored to be classified. Set to None if no need to run the any predictions
pred_dir = r'D:\deep_learning_mehran\data\test'

# Set directories for saving model weights and tensorboard information
cwd_dir = os.getcwd()
ckpt_dir = op.join(cwd_dir, "models_checkpoints")
tboard_dir = op.join(cwd_dir, "tensorboard")
plot_dir = op.join(cwd_dir, "plots")
log_dir = op.join(cwd_dir, "logs")

if not op.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
if not op.isdir(tboard_dir):
    os.mkdir(tboard_dir)
if not op.isdir(plot_dir):
    os.mkdir(plot_dir)
if not op.isdir(log_dir):
    os.mkdir(log_dir)

#######################################################################################################################
# directories to read data and write the results
dir_params = dict(data_dir = data_dir,
                  ckpt_dir = ckpt_dir,
                  tboard_dir = tboard_dir,
                  plot_dir = plot_dir,
                  log_dir = log_dir,
                  pred_dir = pred_dir )

#######################################################################################################################
# create tfrecords
tfrecords_param = dict(shard_size=100, # how many images per tfrecords
                       img_size=(256, 256, 3), # input image size (256, 256, 3)
                       multi_label_csv='train_v2.csv', # set to input csv file name (e.g., train_v2.csv) for multilabel classification and None for binary/multiclass classifications
                       n_classes = None,# number of classes
                       label_to_class=None) # label_to_class can be None if there are too many classes or it is multilabel classification and the code will figure it out!

'''label_to_class={'hemmorhage_data': 0,
                 'non_hemmorhage_data': 1}'''

#######################################################################################################################
# data parameters
data_params = dict(tensor_records=True, # set False if inputs are raw jpg images
                   classification_type='multilabel' , # binary, multiclass, multilabel
                    img_size=(256, 256, 3), # input image size
                    batch_size = 32,
                    augmentation = True, # Set false if no need for augmentation
                    train_split=0.8,
                    val_split=0.1,
                    test_split=0.1,
                    shuffle_size=1000) # best to be equal to the number of images

#######################################################################################################################

# The model can be one of these pretrained models
'''['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 
 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 
 'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 
 'ResNet50V2', 'VGG16', 'VGG19', 'Xception']'''

model_params = dict(resize=False, # set true if it is ok to resize images during training based on the model proper input size
                    optimizer = 'adam', # adam, rmsprop, sgd
                    mixed_precision=False, # True if computations should be done in float16
                    model_name = 'Xception', # The model can be one of the tensorflow application pretrained models listed above
                    weights='imagenet') # imagenet or None (random weights)


#######################################################################################################################

train_params = dict(lr=1e-3, #This is the initial lr. It decreases in each epoch
                    n_epo=10)  # Number of epochs training only top layer



#######################################################################################################################

fine_tune_params = dict(lr=1e-4, # Initial learning rate
                        n_epo=10)

#######################################################################################################################

predict_param = dict(batch_size=100,  # batch size is used to run predictions for batches of images
                     model_name='Xception_fine_tune_min_val_loss.hdf5') # this model is assumed to be in the models_checkpoints folder)

#######################################################################################################################

# identify class labels from folder structure or csv file
if tfrecords_param['label_to_class'] == None and data_params['classification_type'] != 'multilabel':
    lables = []
    for label_name in os.listdir(data_dir):
        if os.path.isdir('/'.join([data_dir, label_name])):
            lables.append(label_name)
    lables.sort()
    label_to_class = {}
    for i, j in enumerate(lables):
        label_to_class[j] = i

    tfrecords_param['label_to_class'] = label_to_class

if tfrecords_param['label_to_class'] == None and data_params['classification_type'] == 'multilabel':
    df_train = pd.read_csv(f"{dir_params['data_dir']}/{tfrecords_param['multi_label_csv']}")

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    label_to_class = {l: i for i, l in enumerate(labels)}
    tfrecords_param['label_to_class'] = label_to_class

if tfrecords_param['n_classes'] == None:
    tfrecords_param['n_classes'] = len(tfrecords_param['label_to_class'])