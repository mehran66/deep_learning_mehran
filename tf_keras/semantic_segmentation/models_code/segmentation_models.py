# Unet, FPN, Linknet, PSPNet (https://github.com/qubvel/segmentation_models)
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

def segmentation_models(pixels, n_classes, activation, backbone_model_name, main_model_name, weights):

    if main_model_name.lower() == 'Unet'.lower():
        base_model = sm.Unet(backbone_name=backbone_model_name.lower(), input_shape=(pixels, pixels, 3), classes=n_classes, activation=activation,
             encoder_weights=weights, encoder_freeze=True,
             decoder_block_type = 'upsampling', decoder_filters = (256, 128, 64, 32, 16), decoder_use_batchnorm = True)

    if main_model_name.capitalize() == 'FPN'.lower():
        base_model = sm.FPN(backbone_name=backbone_model_name.lower(), input_shape=(pixels, pixels, 3), classes=n_classes, activation=activation,
             encoder_weights=weights, encoder_freeze=True,
             pyramid_block_filters=256, pyramid_use_batchnorm=True, pyramid_aggregation='concat', pyramid_dropout=None)

    if main_model_name.capitalize() == 'Linknet'.lower():
        base_model = sm.Linknet(backbone_name=backbone_model_name.lower(), input_shape=(pixels, pixels, 3), classes=n_classes, activation=activation,
             encoder_weights=weights, encoder_freeze=True,
             decoder_block_type='upsampling', decoder_filters=(None, None, None, None, 16), decoder_use_batchnorm=True)

    if main_model_name.capitalize() == 'PSPNet'.lower():
        base_model = sm.PSPNet(backbone_name=backbone_model_name.lower(), input_shape=(pixels, pixels, 3), classes=n_classes, activation=activation,
             encoder_weights=weights, encoder_freeze=True,
             downsample_factor=8, psp_conv_filters=512, psp_pooling_type='avg', psp_use_batchnorm=True, psp_dropout=None)

    return base_model









