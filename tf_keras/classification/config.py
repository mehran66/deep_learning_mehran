import os
from os import path as op


# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Set data directory
data_dir = r'C:\Users\mehra\OneDrive\Desktop\deep_learning\data\BrainTumor'

# Set directories for saving model weights and tensorboard information
cwd_dir = os.getcwd()
ckpt_dir = op.join(cwd_dir, "models")
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
pred_dir = r'C:\Users\mehra\OneDrive\Desktop\deep_learning\data\test\hemmorhage_data'

dir_params = dict(data_dir = data_dir,
                  ckpt_dir = ckpt_dir,
                  tboard_dir = tboard_dir,
                  plot_dir = plot_dir,
                  log_dir = log_dir,
                  pred_dir = pred_dir )


tfrecords_param = dict(shard_size=100, # how many images per tfrecords
                       img_size=(256, 256, 3), # input image size
                       n_classes = 2,
                       label_to_class={
                           'hemmorhage_data': 0,
                           'non_hemmorhage_data': 1}) # label_to_class can be None if there are too many classes and the code will figure it out!

if tfrecords_param['label_to_class'] == None:
    lables = []
    for label_name in os.listdir(data_dir):
        if os.path.isdir('/'.join([data_dir, label_name])):
            lables.append(label_name)
    lables.sort()
    label_to_class = {}
    for i, j in enumerate(lables):
        label_to_class[j] = i

    tfrecords_param['label_to_class'] = label_to_class

data_params = dict(tensor_records=True, # set False if inputs are raw jpg images
                    img_size=(256, 256, 3), # input image size
                    batch_size = 32,
                    augmentation = False, # Set false if no need for augmentation
                    train_split=0.8,
                    val_split=0.1,
                    test_split=0.1)



# The model can be one of these pretrained models
'''['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 
 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2', 
 'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 
 'ResNet50V2', 'VGG16', 'VGG19', 'Xception']'''

model_params = dict(multiclass=False, # False for binary classifications
                    resize=True, # set true if it is ok to resize images during training based on the model proper input size
                    optimizer = 'adam', # adam, rmsprop, sgd
                    mixed_precision=False, # True if computations should be done in float16
                    model_name = 'Xception',
                    weights='imagenet') # imagenet or None (random weights)


train_params = dict(lr=1e-3,
                    n_epo=6)  # Number of epochs training only top layer



fine_tune_params = dict(lr=1e-4,
                        n_epo=6)

predict_param = dict(batch_size=100) # batch size is used to run predictions of batches of images


