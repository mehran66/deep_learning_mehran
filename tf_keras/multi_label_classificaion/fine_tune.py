import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools
import math
import time

import tensorflow as tf
from tensorflow_addons.metrics import FBetaScore


from config import dir_params, tfrecords_param, data_params, model_params, fine_tune_params
from load_data import strg, load_data
from utils import perf_grid

def fine_tune(assess_model = True):

    data_dir = dir_params['data_dir']
    plot_dir = dir_params['plot_dir']
    tboard_dir = dir_params['tboard_dir']
    model_check_points = dir_params['ckpt_dir']

    classification_type = data_params['classification_type']

    epochs = fine_tune_params['n_epo']
    lr = fine_tune_params['lr']

    optimizer = model_params['optimizer']
    mixed_precision = model_params['mixed_precision']

    train_ds, val_ds, test_ds = load_data(assess_data=False)
    n_classes = tfrecords_param['n_classes']

    label_to_class = tfrecords_param['label_to_class']
    class_to_label = {v: k for k, v in label_to_class.items()}

    print("\n######################################################")
    print("read the the config.py file for fine tuning")
    print(f"data directory is: {data_dir}")
    print(f"plot directory is: {plot_dir}")
    print(f"tensorboad directory is: {tboard_dir}")
    print(f"model check points directory is: {model_check_points}")
    print(f"classification type is: {classification_type}")
    print(f"number of epochs is: {epochs}")
    print(f"optimizer is: {optimizer}")
    print(f"learning rate is: {lr}")
    print(f"mixed precision is: {mixed_precision}")
    print(f"number of classes are: {n_classes}")


    strategy, tpu = strg()
    if mixed_precision:
        if tpu:
            mixed_precision.set_global_policy(policy="mixed_bfloat16")
        else:
            mixed_precision.set_global_policy(policy="mixed_float16")
        print('Mixed precision enabled')

    start_time = time.time()

    print("\n######################################################")
    print("#####################FINE TUNING######################")
    print("######################################################")


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
    checkpoint_path = os.path.join(model_check_points, 'fine-tune-min-val_loss.hdf5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor="val_loss", # save the model weights with best validation accuracy
                                                          mode='min',
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
        print(f'leaning rate plot saved in {plot_dir}/fine_tune_learning_rate_scheduler.png')


    show_lr_schedule()

    with strategy.scope():

        if classification_type == 'multiclass':
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics = ["accuracy",
                       FBetaScore(num_classes=n_classes, average='weighted', beta=2.0, threshold=0.5, name='fbeta')]
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

    # load trained model
        loaded_saved_model = tf.keras.models.load_model(model_check_points + r'/training-min-val_loss.hdf5', custom_objects=custom_objects)
        print("Pretrained model was loaded:")
        print(model_check_points + r'/training-min-val_loss.hdf5')

        # Are any of the layers in our model frozen?
        for layer in loaded_saved_model.layers:
          layer.trainable = True # set all layers to trainable
          # print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

        # Check the layers in the base model and see what dtype policy they're using
        # for layer in loaded_saved_model.layers[4].layers:
        #   print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

        # Compile the model
        if optimizer.lower() == 'adam':
            optim = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer.lower() == 'rmsprop':
            optim = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer.lower() == 'sgd':
            optim = tf.keras.optimizers.SGD(learning_rate=lr)

        loaded_saved_model.compile(loss=loss,optimizer=optim,metrics=metrics)

        history_fine_tune = loaded_saved_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback],
            verbose=2
        )

    df_history_fine_tune = pd.DataFrame.from_dict(history_fine_tune.history)
    df_history_fine_tune.to_csv(model_check_points + r'/history-fine-tune.csv',index=False)

    print("\nmodel history were saved in the following file:")
    print(model_check_points + r'/history-fine-tune.csv')

    print(f"\nit took {(time.time() - start_time)} seconds to fine tune the model")

    if assess_model:
        print("\n######################################################")
        print("load and assess the model ...")
        loaded_saved_model = tf.keras.models.load_model(model_check_points + r'/fine-tune-min-val_loss.hdf5', custom_objects=custom_objects)

        # visualize training metrics

        acc = df_history_fine_tune['accuracy']
        val_acc = df_history_fine_tune['val_accuracy']

        loss = df_history_fine_tune['loss']
        val_loss = df_history_fine_tune['val_loss']

        epochs_range = [i+1 for i in range(df_history_fine_tune.shape[0])]

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
        plt.savefig(f'{plot_dir}/fine_tune_step_loss_accuracy_plot.png')
        plt.close()

        print("\nvisualize training metrics. please check!")
        print(f'{plot_dir}/fine_tune_step_loss_accuracy_plot.png')

        # Evaluate model (unsaved version) on whole test dataset
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
            # calculate optimum threshold for each class in a multilabel classification
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
            print(f'ROC AUC plot was saved in {plot_dir}/train_step_ROC_AUC.png')

        if classification_type != 'multilabel':
            def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                plt.figure(figsize=(20, 10))
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
            plt.savefig(f'{plot_dir}/fine_tune_step_confusion_matrix.png')
            plt.close()

            print("\nconfusion matrix using the test data. please check!")
            print(f'{plot_dir}/fine_tune_step_confusion_matrix.png')


        else:
            def confusion_matrix(yt, yp, classes):
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

            confusion_matrix(np.array(y_test), y_pred, np.array(list(label_to_class.keys())))
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
            plt.figure(figsize=(10, 10))
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
                    y_class = y_prob[i] > 0.5

                    mykeys = np.where(y_class == 1)
                    values = [class_to_label[x] for x in mykeys[0]]
                    label_predicted = ', '.join(values)

                    mykeys = np.where(label_batch[i].numpy() == 1)
                    values = [class_to_label[x] for x in mykeys[0]]
                    label_actual = ', '.join(values)

                # display image title with actual and predicted labels
                plt.title(f'Actual: {label_actual},'
                          f'\nPredicted: {label_predicted}')

            plt.savefig(f'{plot_dir}/fine_tune_prediction_on_test_{tmp}.png')
            plt.close()


            print(f'{plot_dir}/fine_tune_prediction_on_test_{tmp}.png')
            tmp += 1

        if classification_type != 'multilabel':
            # find the most 10 confused prediction for each class
            print("\nFind the most confused prediction in each class and visualize up to 10 of them in each class. please check!****\n")
            wrong_pred_indeces = np.where(np.array(y_test) != np.array(y_pred))
            wrong_pred_test = [y_test[i] for i in wrong_pred_indeces[0]]
            wrong_pred_pred = [y_pred[i] for i in wrong_pred_indeces[0]]
            wrong_pred_prob = [y_pred_prob[i] for i in wrong_pred_indeces[0]]

            df = pd.DataFrame(list(zip(wrong_pred_indeces[0], wrong_pred_test, wrong_pred_pred, wrong_pred_prob)),
                              columns=['wrong_pred_indeces', 'wrong_pred_test', 'wrong_pred_pred', 'wrong_pred_prob'])

            for i in range(n_classes):
                df_one_class = df[df['wrong_pred_test'] == i].sort_values(by=['wrong_pred_prob'], ascending=False)

                if df_one_class.shape[0] > 0:

                    fig = plt.figure(figsize=(15, 20))
                    fig.suptitle(class_to_label[i], fontsize=20)
                    j = 1
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
                        j += 1

                    plt.savefig(f'{plot_dir}/fine_tune_most_confused_{class_to_label[i]}.png')
                    plt.close()

                    print(f'{plot_dir}/fine_tune_most_confused_{class_to_label[i]}.png')

                else:
                    print(f'there is not any wrong prediction in {class_to_label[i]}')

if __name__ == "__main__":
    fine_tune(assess_model = True)