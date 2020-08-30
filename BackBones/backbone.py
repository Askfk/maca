"""Build efficient bottom-up networks."""

from BackBones import EfficientNet, ResNet, VGGNet


def build_backbone_net_graph(input_tensor, architecture, config=None):
    """
    Build basic feature extraction networks.
    :param config:
    :param input_tensor: Input of the basic networks, should be a tensor or tf.keras.layers.Input
    :param architecture: The architecture name of the basic network.
    # :param weights: Whether download and initialize weights from the pre-trained weights,
    #                 could be either 'imagenet', (pre-training on ImageNet)
    #                                 'noisy-student',
    #                                 'None' (random initialization)，
    #                                 or the path to the weights file to be loaded。
    :return: Efficient Model and corresponding endpoints.
    """

    if architecture.startswith('efficientnet'):
        model = EfficientNet.EfficientNetX(config, architecture)
        model = model.build_model(input_tensor)
        return model.outputs
    elif architecture.startswith('resnet'):
        model = ResNet.ResNetX(config, architecture)
        model = model.build_model(input_tensor)
        return model.outputs
    elif architecture.startswith('vgg'):
        model = VGGNet.VGGNetX(config, architecture)
        model = model.build_model(input_tensor)
        return model.outputs
    else:
        return [None] * 5


if __name__ == '__main__':
    import tensorflow as tf
    from macacnn_improvement.config import Config
    config_ = Config()
    input_tensor = tf.keras.layers.Input([1024, 1024, 3])
    outputs = build_backbone_net_graph(input_tensor, 'efficientnet-b3', config_)
    model = tf.keras.Model(input_tensor, outputs)
