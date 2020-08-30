"""Build rpn model."""

import tensorflow as tf
from tensorflow import keras


def rpn_graph(shared, anchors_per_location, task):
    """
    Builds the computation graph of Region Proposal Network.

    :param task: 'mask' or 'caption', identification of which task this rpn model is responsible for.
    :param shared: Basic features [batch, height, width, depth]
    :param anchors_per_location: number of anchors per pixel in the feature map
    :param anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                          every pixel in the feature map), or 2 (every other pixel).
    :return: rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
             rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
             rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                       applied to anchors.
    """
    # TODO: Check if stride of 2 causes alignment issues if the feature map is not even

    # Anchor Score. [batch, height, width, anchors per location * 2]
    x = keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear',
                            name='rpn_{}_class_raw'.format(task))(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG
    rpn_probs = keras.layers.Activation("softmax", name="rpn_{}_class".format(task))(rpn_class_logits)

    # Bounding box refinement. [Batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h]
    x = keras.layers.Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
                            activation='linear', name='rpn_{}_bbox_pred'.format(task))(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchors_per_location, depth, task=None):
    """Builds a tensorflow.Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
    """
    input_feature_map = keras.layers.Input(shape=[None, None, depth],
                                           name='input_{}_rpn_feature_map'.format(task))
    outputs = rpn_graph(input_feature_map, anchors_per_location, task)
    return keras.Model([input_feature_map], outputs, name='rpn_{}_model'.format(task))


if __name__ == '__main__':
    model = build_rpn_model(1, 3, 256)