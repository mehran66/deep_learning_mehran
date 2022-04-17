import inspect
import tensorflow as tf

from tensorflow.keras import layers, Model, applications
from tensorflow_addons.metrics import FBetaScore



def model_application(img_size, resize, strategy, model_name, weights, n_classes, optimizer, lr, mix_precision, classification_type):
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
        pixels = model_image_size_map.get(model_name, 224)
        print(f"images are resized to {pixels} to meet the requirements of the model")
    else:
        pixels = img_size[0]

    with strategy.scope():
        base_model = model_dictionary[model_name](input_shape=(pixels, pixels, 3), weights=weights, include_top=False,
                                                  pooling='avg')
        base_model.trainable = False  # freeze base model layers
        base_model.summary()


        # Create Functional model
        inputs1 = layers.Input(shape=img_size, name="input_layer")
        inputs = layers.Resizing(pixels, pixels)(inputs1)
        try:
            inputs = eval('tf.' + model_dictionary[model_name].__module__ + '.preprocess_input')(inputs)
        except:
            inputs = layers.Rescaling(1. / 255)(inputs1)
        x = base_model(inputs, training=False)  # set base_model to inference mode only
        x = layers.Dense(units=1024, activation=tf.nn.relu)(x)
        x = layers.Dropout(0.2)(x)

        if classification_type == 'multiclass':
            if mix_precision:
                x = layers.Dense(n_classes)(x)
                outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
            else:
                outputs = layers.Dense(n_classes, activation="softmax")(x)

        else:
            n_classes = 1 if classification_type == 'binary' else n_classes
            if mix_precision:
                x = layers.Dense(n_classes)(x)
                outputs = layers.Activation("sigmoid", dtype=tf.float32, name="sigmoid_float32")(x)
            else:
                outputs = layers.Dense(n_classes, activation="sigmoid")(x)

        model = Model(inputs1, outputs)


        if optimizer.lower() == 'adam':
            optim = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer.lower() == 'rmsprop':
            optim = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer.lower() == 'sgd':
            optim = tf.keras.optimizers.SGD(learning_rate=lr)

        if classification_type == 'multiclass':
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics = ["accuracy", FBetaScore(num_classes=n_classes, average='weighted', beta=2.0, threshold=0.5, name='fbeta')]
            custom_objects = {'fbeta': FBetaScore}
        elif classification_type == 'binary':
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Recall(name="recall"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.AUC(name='AUC')]
            custom_objects = {}
        elif classification_type == 'multilabel':
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Recall(name="recall"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.AUC(name='AUC'),
                       FBetaScore(num_classes=n_classes, average='weighted', beta=2.0, threshold=0.5, name='fbeta')]
            custom_objects = {'fbeta': FBetaScore}

        # Compile the model
        model.compile(loss=loss, optimizer=optim, metrics=metrics)

    return model, custom_objects

    # ===============================================================================
    # tensorflow hub: it did not perform well
    '''
    import tensorflow_hub as hub

    IMAGE_SIZE = (299, 299)
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        layers.Input(shape=img_size, name="input_layer"),
        tf.keras.layers.Rescaling(1. / 255),
        layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5", trainable=False),
        tf.keras.layers.Dense(n_classes)
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary()
    '''

