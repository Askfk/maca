"""Build detection layer."""

import tensorflow as tf
from tensorflow import keras
from ..macacripts import utils_graph, utils


def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, 1]. positive probs
            deltas: [N, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
                that contains the image excluding the padding.

        Returns detections shaped: [num_detections, (y1, x1, y2, x2, score)] where
            coordinates are normalized.
    """
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    probs = tf.squeeze(probs, axis=1)
    refined_rois = utils_graph.apply_box_deltas_graph(
        rois, deltas * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = utils_graph.clip_boxes_graph(refined_rois, window)

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.compat.v1.where(probs >= 0.15)
        keep = conf_keep[:, 0]

    pre_nms_scores = tf.gather(probs, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)

    bbox_keep = tf.image.non_max_suppression(
        pre_nms_rois, pre_nms_scores,
        max_output_size=config.DETECTION_MAX_INSTANCES,
        iou_threshold=config.DETECTION_NMS_THRESHOLD,
        name='caption_detection_nms')

    # def nms_keep_map(s):
    #     """Apply Non-Maximum Suppression on ROIs of the given class."""
    #     # Apply NMS
    #     class_keep = tf.image.non_max_suppression(
    #         pre_nms_rois, pre_nms_scores,
    #         max_output_size=config.DETECTION_MAX_INSTANCES,
    #         iou_threshold=config.DETECTION_NMS_THRESHOLD,
    #         name='caption_detection_nms')
    #     print('NMS')
    #     print(class_keep.shape)
    #     # Pad with -1 so returned tensors have the same shape
    #     gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=class_keep)[0]
    #     class_keep = tf.pad(tensor=class_keep, paddings=[(0, gap)],
    #                         mode='CONSTANT', constant_values=-1)
    #     print(class_keep.shape)
    #     # Set shape so map_fn() can infer result shape
    #     class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
    #     print(class_keep.shape)
    #
    #     return class_keep
    #
    # # 2. Map over class IDs
    # nms_keep = tf.map_fn(nms_keep_map, pre_nms_scores,
    #                      dtype=tf.int32)
    #
    # print(nms_keep.shape)

    # 3. Merge results into one list, and remove -1 padding
    # nms_keep = tf.reshape(nms_keep, [-1])
    # nms_keep = tf.gather(nms_keep, tf.compat.v1.where(nms_keep > -1)[:, 0])
    # print(nms_keep.shape)
    # keep = tf.expand_dims(nms_keep, 0)
    # keep = tf.sparse.to_dense(keep)[0]
    # print('get here 2')
    # Keep top detections

    roi_count = config.DETECTION_MAX_INSTANCES
    box_scores_keep = tf.gather(pre_nms_scores, bbox_keep)

    num_keep = tf.minimum(tf.shape(box_scores_keep)[0], roi_count)
    # num_keep = tf.minimum(tf.shape(input=box_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(box_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(bbox_keep, top_ids)
    # print('get here 3')
    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(pre_nms_rois, keep),
        tf.gather(pre_nms_scores, keep)[..., tf.newaxis]], axis=1)
    # print('get here 4')
    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(input=detections)[0]
    detections = tf.pad(tensor=detections, paddings=[(0, gap), (0, 0)], mode="CONSTANT")
    # print(detections.shape)
    # print('get here 5')
    return detections


class CaptionDetectionLayer(keras.layers.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
        returns the final detection boxes.

        Returns:
        [batch, num_detections, (y1, x1, y2, x2, class_score)] where
        coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(CaptionDetectionLayer, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(CaptionDetectionLayer, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        rois = inputs[0]
        bbox_scores = inputs[1]
        macacnn_bbox = inputs[2]
        image_meta = inputs[3]

        # print('before into')
        # print(rois.shape)
        # print(bbox_scores.shape)
        # print(mrcnn_bbox.shape)

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = utils_graph.parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = utils_graph.norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, bbox_scores, macacnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 5])

    def compute_output_shape(self, input_shape):
        return [None, self.config.DETECTION_MAX_INSTANCES, 5]

