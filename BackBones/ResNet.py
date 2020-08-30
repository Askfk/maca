import tensorflow as tf


# Reference: https://github.com/brunokovac/Mask-RCNN/blob/master/backbone.py#L4
class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, name, filters, kernel_size, channels_in=None, downsize=False):
        super().__init__()
        if not channels_in:
            channels_in = filters

        c1_strides = (2, 2) if downsize else (1, 1)
        self.c1 = tf.keras.layers.Conv2D(filters, kernel_size, c1_strides, padding="same", name=name + "-conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name=name + "-bn1")
        self.relu1 = tf.keras.layers.Activation("relu")

        self.c2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", name=name + "-conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(name=name + "-bn2")
        self.relu2 = tf.keras.layers.Activation("relu")

        self.shortcut = self.shortcut_method(channels_in, filters, name)

        self.addition = tf.keras.layers.Add()
        self.relu3 = tf.keras.layers.Activation("relu")

        return

    def get_config(self):
        pass

    def call(self, x, training):
        y = self.c1(x)
        y = self.bn1(y, training=training)
        y = self.relu1(y)

        y = self.c2(y)
        y = self.bn2(y, training=training)
        y = self.relu2(y)

        y = self.addition([self.shortcut(x), y])
        y = self.relu3(y)

        return y

    def shortcut_method(self, channels_in, channels_out, name):
        if channels_in != channels_out:
            return tf.keras.layers.Conv2D(channels_out, (1, 1), strides=(2, 2), name=name + "-conv-shortcut")
        else:
            return lambda x: x


class ResNetX(tf.keras.models.Model):

    def __init__(self, config, architecture):
        super().__init__()

        self.config = config

        if architecture == 'resnet50':
            self.resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
        else:
            self.resnet = tf.keras.applications.ResNet101(include_top=False, weights='imagenet')

        self.C2 = self.resnet.get_layer('conv2_block2_out').output
        self.C3 = self.resnet.get_layer('conv3_block4_out').output
        self.C4 = self.resnet.get_layer('conv4_block6_out').output
        self.C5 = self.resnet.get_layer('conv5_block3_out').output
        self.resnet_base = tf.keras.models.Model(self.resnet.input, [self.C2, self.C3, self.C4, self.C5])

        self.P6 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(2, 2), name='fpn_p6')
        self.P5 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p5')
        self.P4 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p4')
        self.P3 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p3')
        self.P2 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='same', name='fpn_p2')

        self.C5P5 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')
        self.C4P4 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')
        self.C3P3 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')
        self.C2P2 = tf.keras.layers.Conv2D(self.config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p2')

        self.P5_UpSampled = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p5upsampled')
        self.P4_UpSampled = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p4Upsampled')
        self.P3_UpSampled = tf.keras.layers.UpSampling2D(size=(2, 2), name='fpn_p3Upsampled')

        self.P4ADD = tf.keras.layers.Add(name='fpn_p4add')
        self.P3ADD = tf.keras.layers.Add(name='fpn_p3add')
        self.P2ADD = tf.keras.layers.Add(name='fpn_p2add')

    def call(self, inputs, training):
        C2, C3, C4, C5 = self.resnet_base(inputs, training=training)

        P5 = self.C5P5(C5)
        P4 = self.P4ADD([self.P5_UpSampled(P5), self.C4P4(C4)])
        P3 = self.P3ADD([self.P4_UpSampled(P4), self.C3P3(C3)])
        P2 = self.P2ADD([self.P3_UpSampled(P3), self.C2P2(C2)])

        P2 = self.P2(P2)
        P3 = self.P3(P3)
        P4 = self.P4(P4)
        P5 = self.P5(P5)
        P6 = self.P6(P5)

        return P2, P3, P4, P5, P6

    def build_model(self, inputs):
        outputs = self.call(inputs, True)
        return tf.keras.models.Model(inputs, outputs, name='resnet_base')


if __name__ == "__main__":
    from config import Config
    resnet50 = ResNetX(Config, 'resnet50')
    inputs = tf.keras.layers.Input([512, 512, 3])
    resnet50 = resnet50.build_model(inputs)
    resnet50.summary()

