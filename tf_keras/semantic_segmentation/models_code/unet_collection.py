from keras_unet_collection import models as kuc_model
# https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection

def unet_collection(pixels, n_classes, activation, backbone_model_name, main_model_name, weights):

    if main_model_name == 'att_unet_2d':
        base_model = kuc_model.att_unet_2d((pixels, pixels, 3), filter_num=[64, 128, 256, 512, 1024], n_labels=n_classes,
                                           stack_num_down=2, stack_num_up=2, activation='ReLU',
                                           atten_activation='ReLU', attention='add',
                                           output_activation=activation.capitalize(),
                                           batch_norm=True, pool=False, unpool=False,
                                           backbone=backbone_model_name, weights=weights,
                                           freeze_backbone=True, freeze_batch_norm=True,
                                           name='attunet')

    elif main_model_name == 'r2_unet_2d':
        base_model = kuc_model.r2_unet_2d((pixels, pixels, 3), filter_num=[64, 128, 256, 512], n_labels=n_classes,
                                          stack_num_down=2, stack_num_up=2, recur_num=2, activation='ReLU',
                                          output_activation=activation.capitalize(),
                                          batch_norm=True, pool=False, unpool=False,
                                          name='r2_unet')

    elif main_model_name == 'resunet_a_2d':
        base_model = kuc_model.resunet_a_2d((pixels, pixels, 3), filter_num=[64, 128, 256, 512],
                                            dilation_num=[1, 3, 15, 31], n_labels=n_classes,
                                            aspp_num_down=256, aspp_num_up=128, activation='ReLU',
                                            output_activation=activation.capitalize(),
                                            batch_norm=True, pool=False, unpool=False,
                                            name='resunet')

    elif main_model_name == 'swin_unet_2d':
        base_model = kuc_model.swin_unet_2d((pixels, pixels, 3), filter_num_begin=128, n_labels=n_classes,
                                            depth=4, stack_num_down=2, stack_num_up=2,
                                            patch_size=(4, 4), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2],
                                            num_mlp=512, output_activation=activation.capitalize(), shift_window=True,
                                            name='swin_unet')

    elif main_model_name == 'transunet_2d':
        base_model = kuc_model.transunet_2d((pixels, pixels, 3), filter_num=[64, 128, 256, 512, 1024], n_labels=n_classes,
                                            embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12, activation='ReLU',
                                            mlp_activation='GELU',
                                            output_activation=activation.capitalize(),
                                            batch_norm=False, pool=True, unpool=True,
                                            backbone=backbone_model_name, weights=weights,
                                            freeze_backbone=True, freeze_batch_norm=True,
                                            name='transunet')

    elif main_model_name == 'u2net_2d':
        base_model = kuc_model.u2net_2d((pixels, pixels, 3), filter_num_down=[64, 128, 256, 512], filter_num_up='auto',
                                        n_labels=n_classes,
                                        filter_mid_num_down='auto', filter_mid_num_up='auto', filter_4f_num='auto',
                                        filter_4f_mid_num='auto',
                                        activation='ReLU',
                                        output_activation=activation.capitalize(),
                                        batch_norm=False, pool=True, unpool=True,
                                        deep_supervision=False,
                                        name='u2net')

    elif main_model_name == 'unet_2d':
        base_model = kuc_model.unet_2d((pixels, pixels, 3), filter_num=[64, 128, 256, 512, 1024], n_labels=n_classes,
                                       stack_num_down=2, stack_num_up=2, activation='ReLU',
                                       output_activation=activation.capitalize(),
                                       batch_norm=True, pool=True, unpool=True,
                                       backbone=backbone_model_name, weights=weights,
                                       freeze_backbone=True, freeze_batch_norm=True,
                                       name='unet')

    elif main_model_name == 'unet_plus_2d':
        base_model = kuc_model.unet_plus_2d((pixels, pixels, 3), filter_num=[64, 128, 256, 512, 1024], n_labels=n_classes,
                                            stack_num_down=2, stack_num_up=2, activation='ReLU',
                                            output_activation=activation.capitalize(),
                                            batch_norm=True, pool=True, unpool=True,
                                            backbone=backbone_model_name, weights=weights,
                                            freeze_backbone=True, freeze_batch_norm=True,
                                            name='xnet')

    elif main_model_name == 'vnet_2d':
        base_model = kuc_model.vnet_2d((pixels, pixels, 3), filter_num=[64, 128, 256, 512], n_labels=n_classes,
                                       res_num_ini=1, res_num_max=3, activation='ReLU',
                                       output_activation=activation.capitalize(),
                                       batch_norm=True, pool=True, unpool=True,
                                       name='vnet')

    else:
        raise ValueError("There is not such model in the unet collection library")

    return base_model