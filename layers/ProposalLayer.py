"""Build proposal layer."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
import numpy as np
from macacripts import utils_graph, utils


class ProposalLayer(keras.layers.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, task, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.task = task
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def get_config(self):
        config = super(ProposalLayer, self).get_config()
        config["config"] = self.config.to_dict()
        config['task'] = self.task
        config["proposal_count"] = self.proposal_count
        config["nms_threshold"] = self.nms_threshold
        return config

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]  # [batch, num_rois]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # print("Proposal Lzyer Check deltas: ", deltas.shape)

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # print("Proposal Layer Check pre_nms_limit: ", pre_nms_limit)
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="{}_top_anchors".format(self.task)).indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.config.IMAGES_PER_GPU,
                                            names=["{}_pre_nms_anchors".format(self.task)])

        # print("Proposal Layer Check ix: ", ix.shape)
        # print("Proposal Layer Check scores: ", scores.shape)
        # print("Proposal Layer Check deltas: ", deltas.shape)
        # print("Proposal Layer Check pre_nms_anchors: ", pre_nms_anchors.shape)

        # Apply deltas to anchors to get refined anchors
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: utils_graph.apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=['{}_refined_anchors'.format(self.task)])

        # print("Proposal Layer Check boxes1: ", boxes.shape)

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: utils_graph.clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["{}_refined_anchors_clipped".format(self.task)])

        # print("Proposal Layer Check boxes2: ", boxes.shape)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
                                                   self.nms_threshold,
                                                   name='{}_rpn_non_max_suppression'.format(self.task))
            # print("Proposal Layer NMS FUNC Check indices: ", indices.shape)
            proposals = tf.gather(boxes, indices)
            scores = tf.gather(scores, indices)
            # print("Proposal Layer NMS FUNC Check proposals1: ", proposals.shape)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            # print("Proposal Layer NMS FUNC Check padding: ", padding)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            scores = tf.pad(scores, [(0, padding)])
            # print("Proposal Layer NMS FUNC Check proposals.shape: ", tf.shape(proposals))
            return proposals, scores

        proposals, scores = utils.batch_slice([boxes, scores], nms,
                                              self.config.IMAGES_PER_GPU)
        # print("Proposal Layer Check proposals: ", proposals.shape)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(None)[0]
            proposals.set_shape(out_shape)
        return [proposals, scores]

    def compute_output_shape(self, input_shape):
        return [
            (None, self.proposal_count, 4),  # proposals
            (None, self.proposal_count)   # scores
        ]
