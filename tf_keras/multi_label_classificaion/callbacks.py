import tensorflow as tf
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math


def callbacks(model_check_points, model_name, tboard_dir, plot_dir, epochs, lr, stage='train'):

    '''

    :param model_check_points:
    :param model_name:
    :param tboard_dir:
    :param plot_dir:
    :param epochs:
    :param lr:
    :param stage: can be train or fine_tune
    :return:
    '''

    # reduce_lr
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5,
                                                     factor=0.2,
                                                     min_delta=1e-2,
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     mode='min',
                                                     min_lr=1e-7)

    # early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=20,
                                                      min_delta=1e-2,
                                                      monitor='val_loss',
                                                      restore_best_weights=True,
                                                      mode='min')

    # Create ModelCheckpoint callback to save model's progress
    checkpoint_path = os.path.join(model_check_points, model_name + f'_{stage}_min_val_loss.hdf5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor="val_loss",
                                                          # save the model weights with best validation loss
                                                          mode='min',
                                                          save_best_only=True,  # only save the best weights
                                                          save_weights_only=False,
                                                          # only save model weights (not whole model)
                                                          verbose=1)  # don't print out whether or not model is being saved

    # tensorboard_callback
    log_dir = tboard_dir + f"/logs-{stage}" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )

    if stage == 'train':
        # learning rate scheduler
        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


        def show_lr_schedule(epochs, lr=1e-3):
            rng = [i for i in range(epochs)]
            y = []
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
            # plt.show()
            plt.savefig(f'{plot_dir}/train_step_learning_rate_scheduler.png')
            plt.close()
            print(f'leaning rate plot saved in {plot_dir}/train_step_learning_rate_scheduler.png')

        show_lr_schedule(epochs=epochs, lr=lr)

    elif stage == 'fine_tune':
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



    return reduce_lr, early_stopping, model_checkpoint, lr_scheduler, tensorboard_callback
