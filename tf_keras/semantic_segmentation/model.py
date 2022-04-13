import inspect
import os
import tensorflow as tf

from tensorflow.keras import layers, Model, applications

from models_code.build_model import build_model

from models_code.unet_collection import unet_collection

from models_code.segmentation_models import segmentation_models

from models_code.loss import define_compile

def model_application(img_size, resize, strategy, backbone_model_name, main_model_name, model_type, weights, multiclass, n_classes, optimizer,
                      lr, mix_precision, loss, mask_to_categorical_flag, freez_mode, freez_batch_norm):

    # https://keras.io/api/applications/
    '''
     ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
      'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S', 'InceptionResNetV2',
      'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50',
      'ResNet50V2', 'VGG16', 'VGG19', 'Xception']
     '''

    model_dictionary = {m[0]: m[1] for m in inspect.getmembers(applications, inspect.isfunction)}

    model_image_size_map = {
        "EfficientNetV2S": 384,
        "EfficientNetV2M": 480,
        "EfficientNetV2L": 480,
        "EfficientNetV2B0": 224,
        "EfficientNetV2B1": 240,
        "EfficientNetV2B2": 260,
        "EfficientNetV2B3": 300,
        "EfficientNetB0": 224,
        "EfficientNetB1": 240,
        "EfficientNetB2": 260,
        "EfficientNetB3": 300,
        "EfficientNetB4": 380,
        "EfficientNetB5": 456,
        "EfficientNetB6": 528,
        "EfficientNetB7": 600,
        "InceptionV3": 299,
        "InceptionResNetV2": 299,
        "NASNetLarge": 331
    }

    if resize:
        pixels = model_image_size_map.get(backbone_model_name, 224)
        print(f"images are resized to {pixels} to meet the requirements of the model")
    else:
        pixels = img_size[0]

    with strategy.scope():

        # Create Functional model

        inputs1 = layers.Input(shape=img_size, name="input_layer")

        try:
            inputs = eval('tf.' + model_dictionary[backbone_model_name].__module__ + '.preprocess_input', name="input_1")(inputs1)
        except:
            inputs = layers.Rescaling(1. / 255, name="input_1")(inputs1)

        activation = 'sigmoid' if n_classes == 1 else 'softmax'

        if model_type == 'build':

            base_model = model_dictionary[backbone_model_name](input_shape=(pixels, pixels, 3), weights=weights, include_top=False, input_tensor=inputs)

            if freez_mode and freez_batch_norm:
                base_model.trainable = False
                for layer in base_model.layers:
                    layer.trainable = False
            elif freez_mode and not freez_batch_norm:
                from tensorflow.keras.layers import BatchNormalization
                for layer in base_model.layers:
                    if isinstance(layer, BatchNormalization):
                        layer.trainable = True
                    else:
                        layer.trainable = False
            else:
                base_model.trainable = True
                for layer in base_model.layers:
                    layer.trainable = True

            outputs = build_model(pixels, n_classes, base_model, activation, inputs, backbone_model_name, main_model_name)

        elif model_type == 'segmentation_models':
            base_model = segmentation_models(pixels, n_classes, activation, backbone_model_name, main_model_name, weights)
            base_model.summary()
            outputs = base_model(inputs)
        elif model_type == 'unet_collection':
            base_model = unet_collection(pixels, n_classes, activation, backbone_model_name, main_model_name, weights)
            base_model.summary()
            outputs = base_model(inputs)
        else:
            raise ValueError("model type does not exist; choose of the options (build, segmentation_models, unet_collection)")

        model = Model(inputs1, outputs)

        metrics, loss, optim, custom_objects = define_compile(multiclass, mask_to_categorical_flag, n_classes, loss, optimizer, lr)

        # Compile the model
        model.compile(loss=loss, optimizer=optim, metrics=metrics)

    return model, custom_objects



