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
from utils import perf_grid
from callbacks import callbacks

def train(assess_model = True):

    print("\n######################################################")
    print("#####################TRAIN MODEL######################")
    print("######################################################")

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']
    tboard_dir = dir_params['tboard_dir']
    model_check_points = dir_params['ckpt_dir']

    classification_type = data_params['classification_type']

    img_size = data_params['img_size']
    epochs = train_params['n_epo']
    optimizer = model_params['optimizer']
    resize = model_params['resize']
    lr = train_params['lr']
    mix_precision = model_params['mixed_precision']
    model_name = model_params['model_name']
    weights = model_params['weights']

    train_ds, val_ds, test_ds = load_data(assess_data=False)
    n_classes = tfrecords_param['n_classes']

    label_to_class = tfrecords_param['label_to_class']
    class_to_label = {v: k for k, v in label_to_class.items()}

    print("\n######################################################")
    print("read the inputs from the config.py file")
    print(f"tensorboad directory is: {tboard_dir}")
    print(f"model check points directory is: {model_check_points}")
    print(f"classification type is: {classification_type}")
    print(f"number of epochs is: {epochs}")
    print(f"optimizer is: {optimizer}")
    print(f"resize is: {resize}")
    print(f"learning rate is: {lr}")
    print(f"mixed precision is: {mix_precision}")
    print(f"model name is: {model_name}")
    print(f"weights is: {weights}")

    strategy, tpu = strg()

    if mix_precision:
        if tpu:
            mixed_precision.set_global_policy(policy="mixed_bfloat16")
        else:
            mixed_precision.set_global_policy(policy="mixed_float16")

        print('Mixed precision enabled')

    start_time = time.time()


    # define callbacks
    reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback = callbacks(model_check_points, model_name, tboard_dir, plot_dir, epochs, lr, stage = 'train')
    print("callbacks were generated: reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback")

    model, custom_objects = model_application(img_size, resize, strategy, model_name, weights, n_classes, optimizer, lr, mix_precision, classification_type)

    # ===============================================================================
    # train model
    print(f"\nmodel {model_name} successfully loaded")

    # print model summary
    model.summary()

    try:
        import pydotplus
        import keras.utils
        keras.utils.vis_utils.pydot = pydotplus
        tf.keras.utils.plot_model(
            model,
            to_file=f'{plot_dir}/train_step_model_{model_name}.png',
            show_shapes=False,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=False,
            dpi=300,
            show_layer_activations=True
        )
        print(f'plot model in: {plot_dir}/train_step_model_{model_name}.png')
    except:
        print("cannot plot the model")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback],
        verbose=2
    )

    print("Model were successfully trained")

    df_history = pd.DataFrame.from_dict(history.history)
    df_history.to_csv(model_check_points + r'/history-train.csv', index=False)

    print("\nmodel history were saved in the following file:")
    print(model_check_points + r'/history.csv')

    print(f"it took {(time.time() - start_time)} seconds to train the model")

    if assess_model:
        print('\n######################################################')
        print("assess model")
        loaded_saved_model = tf.keras.models.load_model(os.path.join(model_check_points, model_name + f'_train_min_val_loss.hdf5'), custom_objects=custom_objects)

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

        print("\nvisualize training metrics. please check!")
        print(f'{plot_dir}/train_step_loss_accuracy_plot.png')

        # Evaluate model (unsaved version) on the whole test dataset
        if str(sum(1 for _ in train_ds)) == 0:
            test_ds = val_ds

        print("\nevaluate model using the test data")
        results_feature_extract_model = loaded_saved_model.evaluate(test_ds, use_multiprocessing=True, return_dict=True, verbose=2)
        results_feature_extract_model

        # ROC and AUC and Confusion matrix
        images, labels = tuple(zip(*test_ds))
        x_test = (np.concatenate([x for x in images], axis=0))
        y_test = list(np.concatenate([y for y in labels], axis=0))
        test_pred = loaded_saved_model.predict(x_test)

        if classification_type == 'multiclass':
            y_pred = [np.argmax(probas) for probas in test_pred]
            y_pred_prob = [max(probas) for probas in test_pred]
        elif classification_type == 'binary':
            y_pred = [int(np.round(probas).tolist()[0]) for probas in test_pred]
            y_pred_prob = [probas.tolist()[0] for probas in test_pred]
        elif classification_type == 'multilabel':
            y_pred = ((test_pred > .05) * 1)
            y_pred_labels = [np.where((probas > 0.5) == 1) for probas in test_pred]
            y_pred_prob = [np.take(probas, cls).tolist()[0] for probas, cls in zip(test_pred, y_pred_labels)]
            pred_label = [np.take(np.array(list(class_to_label.values())), i) for i in y_pred_labels]
            pred_label = [', '.join(i.tolist()[0]) for i in pred_label]


        if classification_type == 'multilabel':
            #calculate optimum threshold for each class in a multilabel classification
            grid = perf_grid(y_pred, np.array(y_test), np.array(list(label_to_class.keys())))
            grid[grid['label'].str.contains('primary')].head(20)
            # Choose the best threshold of
            grid_max = grid.loc[grid.groupby(['id', 'label'])[['f2']].idxmax()['f2'].values]
            pd.set_option('display.max_columns', 100)
            print("\n Accuracy metrics and optimum threshold for each class")
            print(grid_max)

        if classification_type == 'binary':
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

        if classification_type != 'multilabel':
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

            print("\nconfusion matrix using the test data. please check!")
            print(f'{plot_dir}/train_step_confusion_matrix.png')

        else:
            def plot_confusion_matrix(yt, yp, classes):
                instcount = yt.shape[0]
                n_classes = classes.shape[0]
                mtx = np.zeros((n_classes, 4))
                for i in range(instcount):
                    for c in range(n_classes):
                        mtx[c, 0] += 1 if yt[i, c] == 1 and yp[i, c] == 1 else 0
                        mtx[c, 1] += 1 if yt[i, c] == 1 and yp[i, c] == 0 else 0
                        mtx[c, 2] += 1 if yt[i, c] == 0 and yp[i, c] == 0 else 0
                        mtx[c, 3] += 1 if yt[i, c] == 0 and yp[i, c] == 1 else 0
                mtx = [[m0 / (m0 + m1), m1 / (m0 + m1), m2 / (m2 + m3), m3 / (m2 + m3)] for m0, m1, m2, m3 in mtx]
                plt.figure(num=None, figsize=(5, 15), dpi=100, facecolor='w', edgecolor='k')
                plt.imshow(mtx, interpolation='nearest', cmap='Blues')
                plt.title("title")
                tick_marks = np.arange(n_classes)
                plt.xticks(np.arange(4), ['1 - 1', '1 - 0', '0 - 0', '0 - 1'])
                plt.yticks(tick_marks, classes)
                for i, j in itertools.product(range(n_classes), range(4)):
                    plt.text(j, i, round(mtx[i][j], 2), horizontalalignment="center")

                # plt.tight_layout()
                plt.ylabel('labels')
                plt.xlabel('True-Predicted')
                #plt.show()
                plt.savefig(f'{plot_dir}/train_step_confusion_matrix.png')
                plt.close()

            plot_confusion_matrix(np.array(y_test), y_pred, np.array(list(label_to_class.keys())))
            print("\nconfusion matrix using the test data. please check!")
            print(f'{plot_dir}/train_step_confusion_matrix.png')


        # iterate over the test dataset batches and visualize some predictions
        print("\niterate over the test dataset batches and visualize some predictions. please check!")
        tmp = 1
        for dataset in iter(test_ds.take(3)):
            # unpack a single batch of images and labels
            image_batch, label_batch = dataset

            # make predictions on test dataset
            y_prob = loaded_saved_model.predict(image_batch, verbose=1)

            # visualize 10 images from dataset
            plt.figure(figsize=(20, 10))
            for i in range(9):
                # retrieve ith image from current batch and show
                ax = plt.subplot(3, 3, i + 1)
                image = image_batch[i].numpy().astype("uint8")
                plt.imshow(image)
                plt.axis("off")  # turn off axis for clarity

                if classification_type == 'multiclass':
                    y_class = y_prob[i].argmax()
                    label_predicted = class_to_label[y_class]
                    label_actual = class_to_label[label_batch[i].numpy()]
                elif classification_type == 'binary':
                    y_class = int(y_prob[i].round().tolist()[0])
                    label_predicted = class_to_label[y_class]
                    label_actual = class_to_label[label_batch[i].numpy()]
                elif classification_type == 'multilabel':
                    y_class = y_prob[i]>0.5

                    mykeys = np.where(y_class == 1)
                    values = [class_to_label[x] for x in mykeys[0]]
                    label_predicted = ', '.join(values)

                    mykeys = np.where(label_batch[i].numpy() == 1)
                    values = [class_to_label[x] for x in mykeys[0]]
                    label_actual = ', '.join(values)

                # display image title with actual and predicted labels
                plt.title(f'Actual: {label_actual},'
                          f'\nPredicted: {label_predicted}')

            plt.savefig(f'{plot_dir}/train_prediction_on_test_{tmp}.png')
            plt.close()

            print(f'{plot_dir}/train_prediction_on_test_{tmp}.png')
            tmp += 1

        if classification_type != 'multilabel':

            # find the most 10 confused prediction for each class
            print("\nFind the most confused prediction in each class and visualize up to 10 of them in each class. please check!")
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
