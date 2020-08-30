"""Build FPN heads."""

from tensorflow import keras
from layers.BatchNorm import BatchNorm


def build_fpn_bs_graph(rois, pool_size, task, train_bn=False,
                       fc_layers_size=512):
    # TODO: fc_layer_size may need to be changed to reduce params
    """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.

        rois: [batch, num_rois, pool_size, pool_size, channels]
        feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.

        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

        Returns:
            scores: [batch, num_rois, 1]
            bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                         proposal boxes
    """
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size,
                                                         (pool_size, pool_size),
                                                         padding='valid'),
                                     name='macacnn_bs_{}_conv1'.format(task))(rois)
    x = keras.layers.TimeDistributed(BatchNorm(), name='macacnn_bs_{}_bn1'.format(task))(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size, (1, 1)),
                                     name='macacnn_bs_{}_conv2'.format(task))(x)
    x = keras.layers.TimeDistributed(BatchNorm(), name='macacnn_bs_{}_bn2'.format(task))(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)

    shared = keras.layers.Lambda(lambda f: keras.backend.squeeze(
        keras.backend.squeeze(f, 3), 2), name='macacnn_bs_{}_pool_squeeze'.format(task))(x)

    # Generate scores_logits for every bbox, shape = [batch, num_rois, 1]
    scores = keras.layers.TimeDistributed(
        keras.layers.Dense(1), name='macacnn_bs_{}_bbox_scores'.format(task))(shared)

    # BBox head
    # [batch, num_rois, (dy, dx, log(dh), log(dw)]
    maca_bbox = keras.layers.TimeDistributed(
        keras.layers.Dense(4, activation='linear'),
        name='macacnn_bs_{}_bbox_fc'.format(task))(shared)

    return maca_bbox, scores


def fpn_classifier_graph(rois, pool_size, num_classes, train_bn=True,
                         fc_layers_size=512):
    # TODO: fc_layer_size may need to be changed to reduce params
    """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.

        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

        Returns:
            logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
            probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
            bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                         proposal boxes
    """

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size,
                                                         (pool_size, pool_size),
                                                         padding='valid'),
                                     name='macacnn_class_conv1')(rois)
    x = keras.layers.TimeDistributed(BatchNorm(), name='macacnn_class_bn1')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size, (1, 1)),
                                     name='macacnn_class_conv2')(x)
    x = keras.layers.TimeDistributed(BatchNorm(), name='macacnn_class_bn2')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)

    shared = keras.layers.Lambda(lambda f: keras.backend.squeeze(
        keras.backend.squeeze(f, 3), 2), name='pool_squeeze')(x)

    # Classifier head
    macacnn_class_logits = keras.layers.TimeDistributed(
        keras.layers.Dense(num_classes), name='macacnn_class_logits')(shared)

    macacnn_probs = keras.layers.TimeDistributed(
        keras.layers.Activation('softmax'), name='macacnn_class')(macacnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw)]
    x = keras.layers.TimeDistributed(
        keras.layers.Dense(num_classes * 4, activation='linear'),
        name='macacnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw)]
    s = keras.backend.int_shape(x)

    if s[1] == None:
        macacnn_bbox = keras.layers.Reshape((-1, num_classes, 4), name="macacnn_bbox")(x)
    else:
        macacnn_bbox = keras.layers.Reshape((s[1], num_classes, 4), name="macacnn_bbox")(x)

    return macacnn_class_logits, macacnn_probs, macacnn_bbox


def build_fpn_mask_graph(rois, num_classes, train_bn=False):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, num_classes]
    """
    # ROI Pooling
    # Shape [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = rois

    # Conv layers
    for i in range(1, 5):
        x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding='same'),
                                         name='macacnn_mask_conv{}'.format(i))(x)
        x = keras.layers.TimeDistributed(BatchNorm(),
                                         name='macacnn_mask_bn{}'.format(i))(x, training=train_bn)
        x = keras.layers.Activation('relu')(x)

    x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu'),
                                     name='macacnn_mask_deconv')(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'),
                                     name='macacnn_mask')(x)
    return x
