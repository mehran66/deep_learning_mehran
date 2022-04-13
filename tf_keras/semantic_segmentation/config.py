'''
Assumptions
- Images are 3 band RGB in jpg format
- Labels are 1 band image in png format
- Height and width of input images should be divisible by 32 for all models

'''

import os
from os import path as op
import pandas as pd

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Set data directory
data_dir = r'D:\deep_learning\data\building_footprint'

# Set directories for saving model weights and tensorboard information
cwd_dir = os.getcwd()
ckpt_dir = op.join(cwd_dir, "models_trained")
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

# prediction directory where images are stored to be classified
pred_dir_input = r'D:\deep_learning\data\building_footprint\test'
pred_dir_output = r'D:\deep_learning\data\building_footprint\test_binary'


#######################################################################################################################
dir_params = dict(data_dir = data_dir,
                  ckpt_dir = ckpt_dir,
                  tboard_dir = tboard_dir,
                  plot_dir = plot_dir,
                  log_dir = log_dir,
                  pred_dir_input = pred_dir_input,
                  pred_dir_output = pred_dir_output)

#######################################################################################################################
# check 3 things in this step: image/mask format, and size, aw well as mask labels
# the result of preprocessing is RGB images and PNG masks with the identified image/mask size
# also the masks should be a one band raster with unique values for each mask such as 0,1,2,...
# We need to always run this step
# the background (unlabeled) labels should be always placef as the first element in the label_to_class dictionary so the labels zero get assigned to it.
preprocess_parames = dict(input_images_dir_name='images_512', # folder name of input raw images (jpg is the assumed format)
                          input_masks_dir_name='masks_512', # folder name of input raw masks (png is the assumed format)
                          preprocessed_images_dir_name='preprocessed_images', # processed images are saved in this folder
                          preprocessed_masks_dir_name='preprocessed_masks', # processed masks are saved in this folder
                          img_size=(512, 512, 3), # output image size
                          mask_size=(512, 512,1), # output mask size;
                          n_classes=2,
                          label_to_class={'background': 0, 'buildings': 255})

# n_classes =2,
# label_to_class={ 'background': 0, 'buildings': 255}

# n_classes =24,
# label_to_class={ 'unlabeled': [0,0,0], 'paved-area': [128,64,128],
#                  'dirt': [130,76,0], 'grass': [0,102,0],
#                  'gravel': [112,103,87], 'water': [28,42,168],
#                  'rocks': [48,41,30], 'pool': [0,50,89],
#                  'vegetation': [107,142,35], 'roof': [70,70,70],
#                  'wall': [102,102,156], 'window': [254,228,12],
#                  'door': [254,148,12], 'fence': [190,153,153],
#                  'fence-pole': [153,153,153], 'person': [255,22,96],
#                  'dog': [102,51,0], 'car': [9,143,150],
#                  'bicycle': [119,11,32], 'tree': [51,51,0],
#                  'bald-tree': [190,250,190], 'ar-marker': [112,150,146],
#                  'obstacle': [2,135,115], 'conflicting': [255,0,0]}

# n_classes = 2,
# label_to_class = {'unknown': [0, 0, 0], 'urban_land': [0, 255, 255],
#                   'agriculture_land': [255, 255, 0], 'rangeland': [255, 0, 255],
#                   'forest_land': [0, 255, 0], 'water': [0, 0, 255],
#                   'barren_land': [255, 255, 255]})

#######################################################################################################################
# we need to run this step if input data is large and should be patched/tiled
patch_params = dict(images_dir_name_patched='images_patched', # patched images are saved in this folder
                    masks_dir_name_patched='masks_patched', # patched masks are saved in this folder
                    img_size=(620, 620, 3), # patching image size
                    mask_size=(620,620,1), # patching mask size
                    padding = 'SAME') # options are "VALID" (without padding) and "SAME" = with zero padding


#######################################################################################################################

tfrecords_param = dict(ml_images_dir_name = 'preprocessed_images',  # folder name of tiled ML ready images (jpg is the assumed format)
                       ml_masks_dir_name = 'preprocessed_masks',  # folder name of tiled ML ready masks (png is the assumed format)
                       shard_size=100, # how many images per tfrecords
                       img_size=(512, 512, 3), # input image size (if not, images are resized to these values)
                       mask_size=(512, 512, 1))  # input image size (if not, images are resized to these values)


#######################################################################################################################
data_params = dict(tensor_records_dir=data_dir, # set to None if raw images and mask are imported
                    ml_images_dir_name = 'preprocessed_images',  # set to None if tfrecords are imported
                    ml_masks_dir_name = 'preprocessed_masks',  # set to None if tfrecords are imported
                    img_size=(512, 512, 3), # input image size
                    mask_size=(512, 512, 1), # input mask size
                    scale=False, # images/255
                    mask_to_categorical=False, # convert the masks to a one hot encoded raster; set to False for binary classification
                    batch_size = 8,
                    augmentation = False, # Set false if no need for augmentation
                    train_split=0.8,
                    val_split=0.1,
                    test_split=0.1,
                    shuffle_size=100,
                    n_classes = 1, # set to 1 for binary classification
                    label_to_class= 'label_to_class.json')

#######################################################################################################################
# The model can be one of these pretrained models

model_params = dict(multiclass=False, # False for binary classifications
                    resize=False, # set true if it is ok to resize images during training based on the model pretrained input size
                    optimizer = 'adam', # adam, rmsprop, sgd
                    mixed_precision=False, # True if computations should be done in float16
                    model_type = 'segmentation_models', # build, segmentation_models, unet_collection
                    backbone_model_name = 'resnet34',
                    main_model_name = 'Unet',
                    weights='imagenet', # imagenet or None (random weights)
                    loss = 'Crossentropy') # Crossentropy, CategoricalFocalLoss, DiceLoss, CategoricalFocalLoss_DiceLoss

'''
Here are available options for models:

*****1******
model_type = 'build'
backbone_model_name, main_model_name = ResNet50, deeplab_v3_plus
backbone_model_name, main_model_name = InceptionResNetV2, unet
backbone_model_name, main_model_name = MobileNetV2, unet
backbone_model_name, main_model_name = DenseNet121, unet
backbone_model_name, main_model_name = EfficientNetB0, unet
backbone_model_name, main_model_name = VGG16, unet
backbone_model_name, main_model_name = VGG19, unet
backbone_model_name, main_model_name = ResNet50, unet
backbone_model_name, main_model_name = None, multiresunet
backbone_model_name, main_model_name = None, resunet
backbone_model_name, main_model_name = None, unet
backbone_model_name, main_model_name = None, att_unet
backbone_model_name, main_model_name = None, r2_unet
backbone_model_name, main_model_name = None, satellite_unet
backbone_model_name, main_model_name = None, u2net
backbone_model_name, main_model_name = None, xception_unet

*****2******
model_type = 'segmentation_models'
main_model_name can be one of the four options: Unet, FPN, Linknet, PSPNet
backbone_model_name can be: 
    'vgg16' 'vgg19'
    'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
    'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
    'resnext50' 'resnext101'
    'seresnext50' 'seresnext101'
    'senet154'
    'densenet121' 'densenet169' 'densenet201'
    'inceptionv3' 'inceptionresnetv2'
    'mobilenet' 'mobilenetv2'
    'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'

*****3******
model_type = 'unet_collection'
main_model_name that accepts backbone: att_unet_2d, transunet_2d, unet_2d, unet_plus_2d
main_model_name that does not accept backbone: r2_unet_2d, resunet_a_2d, swin_unet_2d, u2net_2d, vnet_2d
supported backbone models are: VGG[16,19], ResNet[50,101,152], ResNet[50,101,152]V2, DenseNet[121,169,201], and EfficientNetB[0-7]

'''

#######################################################################################################################
train_params = dict(lr=1e-3,
                    n_epo=3,  # Number of epochs training only top layer
                    freez_mode = True,
                    freez_batch_norm = True)


#######################################################################################################################
fine_tune_params = dict(lr=1e-4,
                        n_epo=3)
#######################################################################################################################
predict_param = dict(batch_size=100, # batch size is used to run predictions of batches of images
                     model_name='fine_tune_val_loss_min.hdf5', # this model is assumed to be in the models_trained folder
                     binary_threshold=None, # value(0-1)/None/calculate
                     measure=True) # measure the geometrical characteristics of extracted segments


