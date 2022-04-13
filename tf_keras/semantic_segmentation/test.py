import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import shutil

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


ckpt_folder = os.path.join(os.getcwd(), 'ckpt')
if os.path.exists(ckpt_folder):
    shutil.rmtree(ckpt_folder)

ckpt_path = os.path.join(r'D:\deep_learning\tf_keras\semantic_segmentation\logs', 'mymodel_{epoch}')


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath=ckpt_path,
        montior="val_accuracy",
        # save the model weights with best validation accuracy
        mode='max',
        save_best_only=True,  # only save the best weights
        save_weights_only=False,
        # only save model weights (not whole model)
        verbose=1
    )
]

model = get_compiled_model()


model.fit(
    x_train, y_train, epochs=3, batch_size=1, callbacks=callbacks, validation_split=0.2, steps_per_epoch=1
)

