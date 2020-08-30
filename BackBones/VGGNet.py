import tensorflow as tf


class VGGNetX(tf.keras.models.Model):

    def __init__(self, config, architecture):
        super().__init__()

        self.config = config
        if architecture == 'vgg16':
            self.vggnet = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        else:
            self.vggnet = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

        self.C2 = self.vggnet.get_layer('block3_conv3').output
        self.C3 = self.vggnet.get_layer('block4_conv3').output
        self.C4 = self.vggnet.get_layer('block5_conv3').output
        self.C5 = self.vggnet.get_layer('block5_pool').output
        self.vggnet_base = tf.keras.models.Model(self.vggnet.input, [self.C2, self.C3, self.C4, self.C5])

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
        C2, C3, C4, C5 = self.vggnet_base(inputs, training=training)

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


# if __name__ == '__main__':
#     from Project.config import Config
#     inputs = tf.keras.layers.Input([1024, 1024, 3])
#     a = 'vgg16'
#     model = VGGNetX(Config, a)
#     model = model.build_model(inputs)
#     model.summary()