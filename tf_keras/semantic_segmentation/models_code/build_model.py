
def build_model(pixels, n_classes, base_model, activation, inputs, backbone_model_name, main_model_name):

    if main_model_name.lower() == 'deeplab_v3_plus'.lower() and backbone_model_name.lower() == 'ResNet50'.lower():
        from .deeplab3_plus_resnet50 import build_resnet50_deeplab_v3_plus
        outputs = build_resnet50_deeplab_v3_plus(pixels, n_classes, base_model, activation) # ResNet50

    elif main_model_name.lower() == 'unet'.lower() and backbone_model_name.lower() == 'InceptionResNetV2'.lower():
        from .unet_inception_resnet_v2 import build_inception_resnetv2_unet
        outputs = build_inception_resnetv2_unet(n_classes, base_model, activation)  # InceptionResNetV2

    elif main_model_name.lower() == 'unet'.lower() and backbone_model_name.lower() == 'MobileNetV2'.lower():
        from .unet_mobilenet_v2 import build_mobilenetv2_unet
        outputs = build_mobilenetv2_unet(n_classes, base_model, activation) # MobileNetV2

    elif main_model_name.lower() == 'unet'.lower() and backbone_model_name.lower() == 'DenseNet121'.lower():
        from .unet_densnet121 import build_densenet121_unet
        outputs = build_densenet121_unet(n_classes, base_model, activation)  # DenseNet121

    elif main_model_name.lower() == 'unet'.lower() and backbone_model_name.lower() == 'EfficientNetB0'.lower():
        from .unet_efficientnetb0 import build_effienet_unet
        outputs = build_effienet_unet(n_classes, base_model, activation) # EfficientNetB0

    elif main_model_name.lower() == 'unet'.lower() and backbone_model_name.lower() == 'VGG16'.lower():
        from .unet_vgg16 import build_vgg16_unet
        outputs = build_vgg16_unet(n_classes, base_model, activation) # VGG16

    elif main_model_name.lower() == 'unet'.lower() and backbone_model_name.lower() == 'VGG19'.lower():
        from .unet_vgg19 import build_vgg19_unet
        outputs = build_vgg19_unet(n_classes, base_model, activation)  # VGG19

    elif main_model_name.lower() == 'unet'.lower() and backbone_model_name.lower() == 'ResNet50'.lower():
        from .unet_resnet50 import build_resnet50_unet
        outputs = build_resnet50_unet(n_classes, base_model, activation) # ResNet50

    elif backbone_model_name.lower() == None:

        if main_model_name.lower() == 'multiresunet'.lower():
            from .multiresunet import build_multiresunet
            outputs = build_multiresunet(n_classes, inputs, activation)

        elif main_model_name.lower() == 'resunet'.lower():
            from .resunet import build_resunet
            outputs = build_resunet(n_classes, inputs, activation)

        elif main_model_name.lower() == 'unet'.lower():
            from .unet import build_unet
            outputs = build_unet(n_classes, inputs, activation)

        elif main_model_name.lower() == 'att_unet'.lower():
            from .unet_attention import build_att_unet
            outputs = build_att_unet(n_classes, inputs, activation)

        elif main_model_name.lower() == 'r2_unet'.lower():
            from .unet_att_r2 import build_att_r2_unet
            outputs = build_att_r2_unet(n_classes, inputs, activation)

        elif main_model_name.lower() == 'satellite_unet'.lower():
            from .unet_satellite import build_satellite_unet
            outputs = build_satellite_unet(n_classes, inputs, activation)

        elif main_model_name.lower() == 'u2net'.lower():
            from .u2net import build_u2net
            outputs = build_u2net(n_classes, inputs, activation)

        elif main_model_name.lower() == 'xception_unet'.lower():
            from .unet_xception import build_xception_unet
            outputs = build_xception_unet(n_classes, inputs, activation)

        else:
            raise ValueError("There is not such model in the collection")


    return outputs