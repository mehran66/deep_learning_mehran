import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import itertools
import time
import json
import math

import tensorflow as tf
from tensorflow.keras import mixed_precision

from config import dir_params, tfrecords_param, data_params, model_params, train_params, fine_tune_params
from model import model_application
from load_data import strg, load_data
from models_code.loss import define_compile

from skimage.color import label2rgb
from postprocess import measure_mask


from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
import segmentation_models as sm
from models_code.loss import SparseMeanIoU, categorical_focal_loss, categorical_dice_loss, dice_focal_loss


def fine_tune(assess_model=True):
    print("\n######################################################")
    print("read the inputs from the config.py file and load data")

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']
    tboard_dir = dir_params['tboard_dir']
    model_check_points = dir_params['ckpt_dir']

    img_size = data_params['img_size']
    mask_size = data_params['mask_size']

    epochs = fine_tune_params['n_epo']
    optimizer = model_params['optimizer']
    multiclass = model_params['multiclass']
    resize = model_params['resize']
    lr = fine_tune_params['lr']
    mix_precision = model_params['mixed_precision']
    backbone_model_name = model_params['backbone_model_name']
    main_model_name = model_params['main_model_name']
    model_type = model_params['model_type']
    weights = model_params['weights']

    train_ds, val_ds, test_ds = load_data(assess_data=False)

    label_to_class_file = os.path.join(data_dir, data_params['label_to_class'])
    with open(label_to_class_file, 'r') as myfile:
        label_to_class = json.load(myfile)
    n_classes = data_params['n_classes']
    class_to_label = {v: k for k, v in label_to_class.items()}

    loss = model_params['loss']

    mask_to_categorical_flag = data_params['mask_to_categorical']

    freez_mode = train_params['freez_mode']
    freez_batch_norm = train_params['freez_batch_norm']

    print("\n######################################################")
    print("#####################FINE TUNING######################")
    print("######################################################")

    print(f"data directory is: {data_dir}")
    print(f"plot directory is: {plot_dir}")
    print(f"tensorboad directory is: {tboard_dir}")
    print(f"model check points directory is: {model_check_points}")
    print(f"image size is: {img_size}")
    print(f"mask size is: {mask_size}")
    print(f"number of epochs is: {epochs}")
    print(f"optimizer is: {optimizer}")
    print(f"multiclass is: {multiclass}")
    print(f"resize is: {resize}")
    print(f"learning rate is: {lr}")
    print(f"mixed precision is: {mix_precision}")
    print(f"backbone model name is: {backbone_model_name}")
    print(f"main model name is: {main_model_name}")
    print(f"model type name is: {model_type}")
    print(f"weights is: {weights}")
    print(f"number of classes are: {n_classes}")
    print(f"loss function is: {loss}")
    print(f"mask_to_categorical_flag is: {mask_to_categorical_flag}")
    print(f"freez_mode is: {freez_mode}")
    print(f"freez_batch_norm is: {freez_batch_norm}")

    strategy, tpu = strg()

    if mix_precision:
        if tpu:
            mixed_precision.set_global_policy(policy="mixed_bfloat16")
        else:
            mixed_precision.set_global_policy(policy="mixed_float16")

        print('Mixed precision enabled')

    start_time = time.time()

    # define call backs
    #reduce_lr
    reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(patience=5,
                                factor=0.2,
                                min_delta=1e-2,
                                monitor='val_loss',
                                verbose=1,
                                mode='min',
                                min_lr=1e-7)

    #early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                 min_delta=1e-2,
                                  monitor='val_loss',
                                  restore_best_weights=True,
                                  mode='min')


    # Create ModelCheckpoint callback to save model's progress
    checkpoint_path = os.path.join(model_check_points, 'fine_tune_val_IoU_max.hdf5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          montior="val_IoU", # save the model weights with best validation accuracy
                                                          mode='max',
                                                          save_best_only=True, # only save the best weights
                                                          save_weights_only=False, # only save model weights (not whole model)
                                                          verbose=1) # don't print out whether or not model is being saved


    # tensorboard_callback
    log_dir = tboard_dir + "/logs-fine-tune-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )


    '''
    exponential warmup with cosine decay
    The learning rate used is a exponential warmup with cosine decay. 
    The warmup is used to prevent the model from early overfitting on the first images.
    When the model starts learning the loss will be high as the model is trained on ImageNet, not on the training dataset.
    When starting with a high learning rate the model will learn the first few batches very well due to the high loss and could overfit on those samples. 
    When starting with a very low learning rate the model will see all training images and make small adjustment to the weights and therefore learn 
    from all training images equally when the loss is high and weights are modified strongly.
    '''
    def lrfn(epoch, epochs):
        # Config
        LR_START = 1e-6
        LR_MAX = 1e-4
        LR_FINAL = 1e-6
        LR_RAMPUP_EPOCHS = 4
        LR_SUSTAIN_EPOCHS = 0
        DECAY_EPOCHS = epochs  - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        LR_EXP_DECAY = (LR_FINAL / LR_MAX) ** (1 / (epochs - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1))

        if epoch < LR_RAMPUP_EPOCHS: # exponential warmup
            lr = LR_START + (LR_MAX + LR_START) * (epoch / LR_RAMPUP_EPOCHS) ** 2.5
        elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS: # sustain lr
            lr = LR_MAX
        else: # cosine decay
            epoch_diff = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
            decay_factor = (epoch_diff / DECAY_EPOCHS) * math.pi
            decay_factor= (tf.math.cos(decay_factor).numpy() + 1) / 2
            lr = LR_FINAL + (LR_MAX - LR_FINAL) * decay_factor

        return lr

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch, epochs=epochs), verbose=1)


    def show_lr_schedule(epochs=epochs):
        rng = [i for i in range(epochs)]
        y = [lrfn(x, epochs=epochs) for x in rng]
        x = np.arange(epochs)
        x_axis_labels = list(map(str, np.arange(1, epochs + 1)))
        print('init lr {:.1e} to {:.1e} final {:.1e}'.format(y[0], max(y), y[-1]))

        plt.figure(figsize=(10, 10))
        plt.xticks(x, x_axis_labels, fontsize=8)  # set tick step to 1 and let x axis start at 1
        plt.yticks(fontsize=8)
        plt.plot(rng, y)
        plt.grid()
        #plt.show()
        plt.savefig(f'{plot_dir}/fine_tune_learning_rate_scheduler.png')
        plt.close()

    show_lr_schedule()
    print(f'the plot for the learning rate scheduler saved in :{plot_dir}/fine_tune_learning_rate_scheduler.png')
    print("callbacks were generated: reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback")



    print("\n######################################################")
    print("Fine tuning the model ...")
    with strategy.scope():

    # load trained model

        metrics, loss, optim, custom_objects = define_compile(multiclass, mask_to_categorical_flag, n_classes, loss, optimizer, lr)

        loaded_saved_model = tf.keras.models.load_model(model_check_points + r'/train_val_IoU_max.hdf5', custom_objects=custom_objects)
        print("Pretrained model was loaded:")
        print(model_check_points + r'/fine_tune_max_val_IoU.hdf5')

        # Are any of the layers in our model frozen?
        for layer in loaded_saved_model.layers:
          layer.trainable = True # set all layers to trainable
          # print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

        loaded_saved_model.compile(loss=loss, optimizer=optim, metrics=metrics)

        history_fine_tune = loaded_saved_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback],
            verbose=2
        )


    df_history_fine_tune = pd.DataFrame.from_dict(history_fine_tune.history)
    df_history_fine_tune.to_csv(model_check_points + r'/history-fine-tune.csv', index=False)

    print("model was fine tuned and its history were saved in the following file:")
    print(model_check_points + r'/history-fine-tune.csv')

    print(f"it took {(time.time() - start_time)} seconds to fine tune the model")

    if assess_model:
        print("\n######################################################")
        print("load and assess the model ...")
        loaded_saved_model = tf.keras.models.load_model(model_check_points + r'/fine_tune_val_IoU_max.hdf5', custom_objects=custom_objects)

        # visualize training metrics
        acc = df_history_fine_tune['accuracy']
        val_acc = df_history_fine_tune['val_accuracy']

        loss = df_history_fine_tune['loss']
        val_loss = df_history_fine_tune['val_loss']

        iou = df_history_fine_tune['IoU']
        val_iou = df_history_fine_tune['val_IoU']

        epochs_range = [i + 1 for i in range(df_history_fine_tune.shape[0])]

        plt.figure(figsize=(15, 10))
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, iou, label='Training IoU')
        plt.plot(epochs_range, val_iou, label='Validation IoU')
        plt.legend(loc='upper right')
        plt.title('Training and Validation IoU')
        # plt.show()
        plt.savefig(f'{plot_dir}/fine-tune_loss_accuracy_plot.png')
        plt.close()

        print("visualize training metrics. please check!")
        print(f'{plot_dir}/fine-tune_loss_accuracy_plot.png')

        # Evaluate model (unsaved version) on whole test dataset
        if str(sum(1 for _ in train_ds)) == 0:
            test_ds = val_ds

        print("\nevaluate model using the test data")
        results_feature_extract_model = loaded_saved_model.evaluate(test_ds, use_multiprocessing=True, return_dict=True, verbose=2)
        results_feature_extract_model


        # metrics and plots

        images, labels = tuple(zip(*test_ds))
        x_test = (np.concatenate([x for x in images], axis=0))
        y_test = (np.concatenate([y for y in labels], axis=0))


        test_pred = loaded_saved_model.predict(x_test)
        if multiclass:
            y_test_argmax = np.argmax(y_test, axis=3)
            y_pred_argmax = np.argmax(test_pred, axis=3)
        else:
            y_test_argmax = y_test
            y_pred_argmax = np.where(test_pred > 0.5, 1, 0)


        # Using built in keras function for IoU
        from keras.metrics import MeanIoU
        num_classes = 2 if n_classes == 1 else n_classes
        IOU_keras = MeanIoU(num_classes=num_classes)
        IOU_keras.update_state(y_test_argmax, y_pred_argmax)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)

        # To calculate I0U for each class...
        for i in range(n_classes):
            tmp = values[i,i] / (values[i,:].sum() + values[:,i].sum() - values[i,i])
            print(f"IoU of class {class_to_label[i]} is = {tmp}")

        def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(10, 10))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                pass

            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            pass

        np.set_printoptions(precision=2)

        plt.figure()
        plot_confusion_matrix(values, classes=list(label_to_class.keys()))
        plt.savefig(f'{plot_dir}/fine_tune_step_confusion_matrix.png')
        plt.close()
        print("confusion matrix using the test data. please check!****\n")
        print(f'{plot_dir}/fine_tune_step_confusion_matrix.png')

        # iterate over the test dataset batches and visualize some predictions
        print("\niterate over the test dataset batches and visualize some predictions. please check!")
        tmp = 1
        for dataset in iter(test_ds.take(10)):
            # unpack a single batch of images and labels
            image_batch, label_batch = dataset

            test_pred = loaded_saved_model.predict(image_batch)
            if multiclass:
                y_test = [np.argmax(probas, axis=2) for probas in label_batch]
                y_pred = [np.argmax(probas, axis=2) for probas in test_pred]
                label_images = [measure_mask(image, mask, probability=False) for image, mask in
                                zip(image_batch, y_pred)]
            else:
                y_test = label_batch
                y_pred = test_pred
                label_images = [measure_mask(np.squeeze(image), np.squeeze(mask), probability=True) for image, mask in
                                zip(image_batch, y_pred)]

            n_examples = 3
            fig, axs = plt.subplots(n_examples, 3, figsize=(14, n_examples * 7), constrained_layout=True)
            for ax, ele in zip(axs, range(n_examples)):
                image = image_batch[ele]
                y_true = y_test[ele]
                label_image = label_images[ele][1]
                prediction = label2rgb(label_image, image=image.numpy().astype(np.uint8), saturation=1)
                ax[0].set_title('Original image')
                ax[0].imshow(image / 255)
                ax[1].set_title('Original mask')
                ax[1].imshow(np.squeeze(y_true))
                ax[2].set_title('Predicted area')
                ax[2].imshow(prediction)

            plt.savefig(f'{plot_dir}/fine_tune_prediction_on_test_{tmp}.png')
            plt.close()

            print(f'{plot_dir}/fine_tune_prediction_on_test_{tmp}.png')
            tmp += 1

if __name__ == "__main__":
    fine_tune(assess_model=True)