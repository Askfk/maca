import tensorflow as tf
from tensorflow import keras

from ..macacripts import utils_graph, utils
from .MaskDetectionTargetLayer import overlaps_graph, trim_zeros_graph


def detection_targets_graph(proposals, gt_boxes, gt_captions, gt_scores, config):
    """

    :param gt_scores: [N]
    :param proposals: [N, 4]
    :param gt_boxes: [N, 4]
    :param gt_captions: [N, MAX_LENGTH]
    :param config:
    :return:
    """

    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name='roi_assertion')]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name='trim_proposals')
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name='trim_gt_boxes')
    # print(gt_captions.shape)
    # gt_captions = tf.boolean_mask(gt_captions, non_zeros, name='trim_gt_captions')
    gt_captions = tf.gather(gt_captions, tf.compat.v1.where(non_zeros)[:, 0], axis=0,
                            name='trim_gt_captions')
    gt_scores = tf.gather(gt_scores, tf.compat.v1.where(non_zeros)[:, 0], axis=0,
                          name='trim_gt_scores')

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(input_tensor=overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.compat.v1.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    positive_scores = tf.gather(gt_scores, positive_indices)
    negative_scores = tf.zeros_like(negative_indices, tf.float32)

    # Assign positive ROIs to GT boxes
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        pred=tf.greater(tf.shape(input=positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64))
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils_graph.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to captions
    roi_gt_captions = tf.gather(gt_captions, roi_gt_box_assignment)

    rois = tf.concat([positive_rois, negative_rois], axis=0)
    roi_gt_scores = tf.concat([positive_scores, negative_scores], axis=0)
    N = tf.shape(input=negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
    rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
    deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])
    roi_gt_captions = tf.pad(tensor=roi_gt_captions, paddings=[(0, N + P), (0, 0)])
    roi_gt_scores = tf.pad(tensor=roi_gt_scores, paddings=[(0, P)], constant_values=-1)

    return rois, deltas, roi_gt_captions, roi_gt_scores


class CaptionDetectionTargetLayer(keras.layers.Layer):
    """

    """
    def __init__(self, config, **kwargs):
        super(CaptionDetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(CaptionDetectionTargetLayer, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        proposals = inputs[0]
        gt_boxes = inputs[1]
        gt_captions = inputs[2]
        gt_scores = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["caption_rois", "caption_target_deltas", "target_caption", "target_caption_scores"]
        outputs = utils.batch_slice(
            [proposals, gt_boxes, gt_captions, gt_scores],
            lambda x, y, z, w: detection_targets_graph(x, y, z, w, self.config),
            self.config.IMAGES_PER_GPU, names=names)

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MAX_LENGTH),  # captions
            (None, self.config.TRAIN_ROIS_PER_IMAGE)   # scores
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]
