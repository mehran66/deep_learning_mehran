import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
import segmentation_models as sm


# https://github.com/artemmavrin/focal-loss

# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#          Focal loss          #
################################
# https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
# this cannot be used with sparse categorical
def categorical_focal_loss(alpha=0.25, beta=None, gamma_f=2.):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """

    def loss_function(y_true, y_pred):

        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss

    return loss_function


def binary_focal_loss(targets, inputs, alpha=0.25, gamma=2.0):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss


################################
#           Dice loss          #
################################
def categorical_dice_loss(delta=0.5, smooth=0.000001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        dice_loss = K.mean(1 - dice_class)

        return dice_loss

    return loss_function


def binary_dice_loss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def binary_dice_bce_loss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

################################
#           Dice+foccal loss          #
################################
def dice_focal_loss(alpha=0.25, beta=None, gamma_f=2., delta=0.5, smooth=0.000001):

    def loss_function(y_true, y_pred):

        # focal loss
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))

        # dice loss
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1 - y_pred), axis=axis)
        fp = K.sum((1 - y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        dice_loss = K.mean(1 - dice_class)

        return focal_loss + dice_loss

    return loss_function

################################
#           Metrics          #
################################

# MeanIoU for sprse masks
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(SparseMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)


def binary_IoULoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU



'''
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function
'''

################################
#           define_compile          #
################################

def define_compile(multiclass, mask_to_categorical_flag, n_classes, loss, optimizer, lr):

    if multiclass:
        if mask_to_categorical_flag:

            metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                       tf.keras.metrics.OneHotMeanIoU(n_classes, name="IoU")]  # sm.metrics.iou_score

            if loss == 'Crossentropy':
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                custom_objects = {}


            elif loss == 'CategoricalFocalLoss':
                loss = categorical_focal_loss()
                custom_objects = {'loss_function': categorical_focal_loss()}

            elif loss == 'DiceLoss':
                loss = categorical_dice_loss()
                custom_objects = {'loss_function': categorical_dice_loss()}

            elif loss == 'CategoricalFocalLoss_DiceLoss':
                loss = dice_focal_loss()
                custom_objects = {'loss_function': dice_focal_loss()}

        else:
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                       SparseMeanIoU(num_classes=n_classes, name='IoU')]

            if loss == 'Crossentropy':
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                custom_objects = {'SparseMeanIoU': SparseMeanIoU}

            elif loss == 'CategoricalFocalLoss':
                loss = SparseCategoricalFocalLoss(gamma=2, from_logits=False)
                custom_objects = {'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss, 'SparseMeanIoU': SparseMeanIoU}

    else:
        metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                   tf.keras.metrics.Recall(name="recall"),
                   tf.keras.metrics.Precision(name="precision"),
                   tf.keras.metrics.AUC(name='AUC'),
                   tf.keras.metrics.BinaryIoU(name='IoU')]

        if loss == 'Crossentropy':
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            custom_objects = {}

        elif loss == 'CategoricalFocalLoss':
            loss = BinaryFocalLoss(gamma=2, from_logits=False)
            custom_objects = {'BinaryFocalLoss': BinaryFocalLoss}


    if optimizer.lower() == 'adam':
        optim = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer.lower() == 'rmsprop':
        optim = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer.lower() == 'sgd':
        optim = tf.keras.optimizers.SGD(learning_rate=lr)

    return metrics, loss, optim, custom_objects