import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools
import time


import tensorflow as tf
from tensorflow.keras import mixed_precision

from config import dir_params, tfrecords_param, data_params, model_params, train_params
from model import model_application
from load_data import strg, load_data

def train(assess_model = True):

    print("read the inputs from the config.py file")

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']
    tboard_dir = dir_params['tboard_dir']
    model_check_points = dir_params['ckpt_dir']

    img_size = data_params['img_size']
    epochs = train_params['n_epo']
    optimizer = model_params['optimizer']
    multiclass = model_params['multiclass']
    resize = model_params['resize']
    lr = train_params['lr']
    mix_precision = model_params['mixed_precision']
    model_name = model_params['model_name']
    weights = model_params['weights']

    train_ds, val_ds, test_ds = load_data(assess_data=False)
    n_classes = tfrecords_param['n_classes']

    label_to_class = tfrecords_param['label_to_class']
    class_to_label = {v: k for k, v in label_to_class.items()}

    print(f"data directory is: {data_dir}")
    print(f"plot directory is: {plot_dir}")
    print(f"tensorboad directory is: {tboard_dir}")
    print(f"model check points directory is: {model_check_points}")
    print(f"image size is: {img_size}")
    print(f"number of epochs is: {epochs}")
    print(f"optimizer is: {optimizer}")
    print(f"multiclass is: {multiclass}")
    print(f"resize is: {resize}")
    print(f"learning rate is: {lr}")
    print(f"mixed precision is: {mix_precision}")
    print(f"model name is: {model_name}")
    print(f"weights is: {weights}")
    print(f"number of classes are: {n_classes}")

    strategy, tpu = strg()

    if mix_precision:
        if tpu:
            mixed_precision.set_global_policy(policy="mixed_bfloat16")
        else:
            mixed_precision.set_global_policy(policy="mixed_float16")

        print('Mixed precision enabled')

    start_time = time.time()

    # define callbacks

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
    checkpoint_path = os.path.join(model_check_points, 'training-min-val_loss.hdf5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          montior="val_loss", # save the model weights with best validation accuracy
                                                          mode='min',
                                                          save_best_only=True, # only save the best weights
                                                          save_weights_only=False, # only save model weights (not whole model)
                                                          verbose=1) # don't print out whether or not model is being saved

    # learning rate scheduler
    def scheduler(epoch, lr):
      if epoch < 5:
        return lr
      else:
        return lr * tf.math.exp(-0.1)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    def show_lr_schedule(epochs=epochs, lr = 1e-3):
        rng = [i for i in range(epochs)]
        y=[]
        for x in rng:
            lr = scheduler(x, lr)
            y.append(lr)
        x = np.arange(epochs)
        x_axis_labels = list(map(str, np.arange(1, epochs + 1)))
        print('init lr {:.1e} to {:.1e} final {:.1e}'.format(y[0], max(y), y[-1]))

        plt.figure(figsize=(10, 10))
        plt.xticks(x, x_axis_labels, fontsize=8)  # set tick step to 1 and let x axis start at 1
        plt.yticks(fontsize=8)
        plt.plot(rng, y)
        plt.grid()
        #plt.show()
        plt.savefig(f'{plot_dir}/train_step_learning_rate_scheduler.png')
        plt.close()

    show_lr_schedule(lr = 1e-3)

    # tensorboard_callback
    log_dir = tboard_dir + "/logs-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )

    print("***callbacks were generated: reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback")

    model = model_application(img_size, resize, strategy, model_name, weights, multiclass, n_classes, optimizer, lr, mix_precision)

    # ===============================================================================
    # train model
    print(f"*** model {model_name} successfully loaded")

    # print model summary
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback],
        verbose=2
    )

    print("*** model were successfully trained")

    df_history = pd.DataFrame.from_dict(history.history)
    df_history.to_csv(model_check_points + r'/history-train.csv', index=False)

    print("*** model history were saved in the following file:")
    print(model_check_points + r'/history.csv')

    print(f"it took {(time.time() - start_time)} seconds to train the model")

    if assess_model:
        loaded_saved_model = tf.keras.models.load_model(model_check_points + r'/training-min-val_loss.hdf5')

        # visualize training metrics
        acc = df_history['accuracy']
        val_acc = df_history['val_accuracy']

        loss = df_history['loss']
        val_loss = df_history['val_loss']

        epochs_range = [i+1 for i in range(df_history.shape[0])]

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        #plt.show()
        plt.savefig(f'{plot_dir}/train_step_loss_accuracy_plot.png')
        plt.close()

        print("visualize training metrics. please check!****\n")
        print(f'{plot_dir}/train_step_loss_accuracy_plot.png')

        # Evaluate model (unsaved version) on whole test dataset
        if str(sum(1 for _ in train_ds)) == 0:
            test_ds = val_ds

        print("****evaluate model using the test data")
        results_feature_extract_model = loaded_saved_model.evaluate(test_ds, verbose=2)
        results_feature_extract_model


        # ROC and AUC and Confusion matrix

        images, labels = tuple(zip(*test_ds))
        x_test = (np.concatenate([x for x in images], axis=0))
        y_test = list(np.concatenate([y for y in labels], axis=0))
        test_pred = loaded_saved_model.predict(x_test)

        if multiclass:
            y_pred = [np.argmax(probas) for probas in test_pred]
            y_pred_prob = [max(probas) for probas in test_pred]
        else:
            y_pred = [int(np.round(probas).tolist()[0]) for probas in test_pred]
            y_pred_prob = [probas.tolist()[0] for probas in test_pred]

        if not multiclass:
            # plot ROC and AUC
            fp, tp, thresholds = roc_curve(y_test, test_pred)
            auc_keras = auc(fp, tp)

            plt.figure(figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fp, tp, label='AUC (area = {:.3f})'.format(auc_keras))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')

            plt.grid()
            skip = 1
            for i in range(0, len(thresholds), skip):
                plt.text(fp[i], tp[i], thresholds[i])

            #plt.show()
            plt.savefig(f'{plot_dir}/train_step_ROC_AUC.png')
            plt.close()

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

        cnf_mat = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        plt.figure()
        plot_confusion_matrix(cnf_mat, classes=list(label_to_class.keys()))
        #plt.show()
        plt.savefig(f'{plot_dir}/train_step_confusion_matrix.png')
        plt.close()

        print("confusion matrix using the test data. please check!****\n")
        print(f'{plot_dir}/train_step_confusion_matrix.png')


        # iterate over the test dataset batches and visualize some predictions
        print("iterate over the test dataset batches and visualize some predictions. please check!****\n")
        tmp = 1
        for dataset in iter(test_ds.take(3)):
            # unpack a single batch of images and labels
            image_batch, label_batch = dataset

            # make predictions on test dataset
            y_prob = loaded_saved_model.predict(image_batch, verbose=1)

            # visualize 10 images from dataset
            plt.figure(figsize=(10, 10))
            for i in range(9):
                # retrieve ith image from current batch and show
                ax = plt.subplot(3, 3, i + 1)
                image = image_batch[i].numpy().astype("uint8")
                plt.imshow(image)
                plt.axis("off")  # turn off axis for clarity

                # index of the highest probability indicates predicted class
                if multiclass:
                    y_class = y_prob[i].argmax()
                else:
                    y_class = int(y_prob[i].round().tolist()[0])


                # display image title with actual and predicted labels
                plt.title(f'Actual: {class_to_label[label_batch[i].numpy()]},'
                          f'\nPredicted: {class_to_label[y_class]}')

            plt.savefig(f'{plot_dir}/train_prediction_on_test_{tmp}.png')
            plt.close()

            print(f'{plot_dir}/train_prediction_on_test_{tmp}.png')
            tmp += 1

        # find the most 10 confused prediction for each class
        print("Find the most confused prediction in each class and visualize up to 10 of them in each class. please check!****\n")
        wrong_pred_indeces = np.where(np.array(y_test) != np.array(y_pred))
        wrong_pred_test = [y_test[i] for i in wrong_pred_indeces[0]]
        wrong_pred_pred = [y_pred[i] for i in wrong_pred_indeces[0]]
        wrong_pred_prob = [y_pred_prob[i] for i in wrong_pred_indeces[0]]

        df = pd.DataFrame(list(zip(wrong_pred_indeces[0], wrong_pred_test, wrong_pred_pred, wrong_pred_prob)),
                          columns=['wrong_pred_indeces', 'wrong_pred_test', 'wrong_pred_pred', 'wrong_pred_prob'])


        for i in range(n_classes):
            df_one_class = df[df['wrong_pred_test'] == i].sort_values(by=['wrong_pred_prob'], ascending=False)

            if df_one_class.shape[0]>0:

                fig = plt.figure(figsize=(15, 20))
                fig.suptitle(class_to_label[i], fontsize=20)
                j=1
                for row in df_one_class.head(9).iterrows():
                    index = row[1]['wrong_pred_indeces']
                    ax = plt.subplot(3, 3, j)
                    image = x_test[int(index)].astype("uint8")
                    plt.imshow(image)
                    plt.axis("off")  # turn off axis for clarity

                    # display image title with actual and predicted labels
                    plt.title(f"Actual: {class_to_label[int(row[1]['wrong_pred_test'])]},"
                              f"\nPredicted: {class_to_label[int(row[1]['wrong_pred_pred'])]},"
                              f"\nProbability: {row[1]['wrong_pred_prob']}")
                    j+=1

                plt.savefig(f'{plot_dir}/train_most_confused_{class_to_label[i]}.png')
                plt.close()

                print(f'{plot_dir}/train_most_confused_{class_to_label[i]}.png')

            else:
                print(f'there is not any wrong prediction in {class_to_label[i]}')

if __name__ == "__main__":
    train(assess_model = True)
