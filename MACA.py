import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
import re
from collections import OrderedDict

from .BackBones.backbone import build_backbone_net_graph

from .layers.ROIAlignLayer import ROIAlign
from .layers.RPNModel import build_rpn_model
from .layers.ProposalLayer import ProposalLayer
from .layers.FPNHeads import build_fpn_bs_graph, fpn_classifier_graph, build_fpn_mask_graph
from .layers.MaskDetectionTargetLayer import MaskDetectionTargetLayer
from .layers.CaptionDetectionTargetLayer import CaptionDetectionTargetLayer
from .layers.CaptionLayer import build_caption_layer_graph
from .layers.CaptionDetectionLayer import CaptionDetectionLayer
from .layers.MaskDetectionLayer import MaskDetectionLayer

from .macacripts.DataGenerator import DataGenerator
from .macacripts import losses, utils_graph, utils

tf.compat.v1.disable_eager_execution()


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("Shape: {:20}   ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}   max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}   max: {:10}".format("", ""))

        text += "  {}".format(array.dtype)
    print(text)


# A hack to get around Keras's bad support for constants
# This class returns a constant layer
class ConstLayer(tf.keras.layers.Layer):
    def __init__(self, x, name=None):
        super(ConstLayer, self).__init__(name=name)
        self.x = tf.Variable(x)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        return self.x


class MACA():

    def __init__(self, mode, config, model_dir, tokenizer=None):

        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.tokenizer = tokenizer or config.TOKENIZER
        self.model_dir = model_dir
        self.set_log_dir()
        self.model = self.build(mode=mode, config=config)
        self.exists_loss = []
        self.pre_frame_detection = None

    def build(self, mode, config):
        assert mode in ['training', 'inference']

        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = keras.layers.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name='input_image')
        input_image_meta = keras.layers.Input(shape=[config.IMAGE_META_SIZE],
                                              name='input_image_meta')

        if mode == 'training':
            # RPN GT
            # 1. Masking Part
            input_mask_rpn_match = keras.layers.Input(
                shape=[None, 1], name='input_mask_rpn_match', dtype=tf.int32)
            input_mask_rpn_bbox = keras.layers.Input(
                shape=[None, 4], name='input_mask_rpn_bbox', dtype=tf.float32)

            # 2. Captioning Part
            input_caption_rpn_match = keras.layers.Input(
                shape=[None, 1], name='input_caption_rpn_match', dtype=tf.int32)
            input_caption_rpn_bbox = keras.layers.Input(
                shape=[None, 4], name='input_rpn_rpn_bbox', dtype=tf.float32)

            input_gt_class_ids = keras.layers.Input(
                shape=[None], name='input_gt_class_ids', dtype=tf.int32)

            # Detection GT (captions, bounding boxes, masks)
            # 1. Mask Part (Zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = keras.layers.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name='input_gt_masks', dtype=bool)
            else:
                input_gt_masks = keras.layers.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name='input_gt_masks', dtype=bool)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_mask_gt_boxes = keras.layers.Input(
                shape=[None, 4], name='input_mask_gt_boxes', dtype=tf.float32)
            # Normalize coordinates
            mask_gt_boxes = keras.layers.Lambda(
                lambda x: utils_graph.norm_boxes_graph(
                    x, keras.backend.shape(input_image)[1:3]))(input_mask_gt_boxes)

            # 2. Caption Part
            input_gt_captions = keras.layers.Input(
                shape=[None, config.MAX_LENGTH], name='input_gt_captions', dtype=tf.int32)
            input_caption_gt_boxes = keras.layers.Input(
                shape=[None, 4], name='input_caption_gt_boxes', dtype=tf.float32)
            input_caption_gt_boxes_scores = keras.layers.Input(
                shape=[None], name='input_caption_gt_boxes_scores', dtype=tf.float32)
            # Normalize coordinates
            caption_gt_boxes = keras.layers.Lambda(
                lambda x: utils_graph.norm_boxes_graph(
                    x, keras.backend.shape(input_image)[1:3]))(input_caption_gt_boxes)

        elif mode == 'inference':
            input_anchors = keras.layers.Input(
                shape=[None, 4], name='input_anchors')

        # Build the shared convolutional layers.
        P2, P3, P4, P5, P6 = build_backbone_net_graph(input_image, self.config.BACKBONE, self.config)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        macacnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == 'training':
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

            # A hack to get around Keras's bad support for constants
            # This class returns a constant layer
            anchors = ConstLayer(anchors, name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn_mask = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), 512, 'mask')
        rpn_caption = build_rpn_model(len(config.RPN_ANCHOR_RATIOS), 512, 'caption')
        # Loop through pyramid layers
        mask_layer_outputs = []
        caption_layer_outputs = []
        # TODO: Check if stride of 2 causes alignment issues if the feature map is not even

        # Shared convolutional base of the RPN
        shared_layer = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                                           strides=config.RPN_ANCHOR_STRIDE,
                                           name='rpn_conv_shared')
        for p in rpn_feature_maps:
            shared = shared_layer(p)
            mask_layer_outputs.append(rpn_mask([shared]))
            caption_layer_outputs.append(rpn_caption([shared]))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        mask_output_names = ["mask_rpn_class_logits", "mask_rpn_class", "mask_rpn_bbox"]
        caption_output_names = ["caption_rpn_class_logits", "caption_rpn_class", "caption_rpn_bbox"]

        mask_outputs = list(zip(*mask_layer_outputs))
        caption_outputs = list(zip(*caption_layer_outputs))

        mask_outputs = [keras.layers.Concatenate(axis=1, name=n)(list(o))
                        for o, n in zip(mask_outputs, mask_output_names)]
        caption_outputs = [keras.layers.Concatenate(axis=1, name=n)(list(o))
                           for o, n in zip(caption_outputs, caption_output_names)]

        mask_rpn_class_logits, mask_rpn_class, mask_rpn_bbox = mask_outputs
        caption_rpn_class_logits, caption_rpn_class, caption_rpn_bbox = caption_outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE

        proposal_layer = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            task='mask',
            config=config,
            name="ROI")
        mask_rpn_rois, mask_roi_scores = proposal_layer([mask_rpn_class, mask_rpn_bbox, anchors])
        caption_rpn_rois, caption_roi_scores = proposal_layer([caption_rpn_class, caption_rpn_bbox, anchors])

        if mode == 'training':
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = keras.layers.Lambda(
                lambda x: utils_graph.parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                mask_input_rois = keras.layers.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                                     name="mask_input_roi", dtype=np.int32)
                caption_input_rois = keras.layers.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                                        name="caption_input_roi", dtype=np.int32)

                # Normalize coordinates
                mask_target_rois = keras.layers.Lambda(
                    lambda x: utils_graph.norm_boxes_graph(
                        x, keras.backend.shape(input_image)[1:3]))(mask_input_rois)
                caption_target_rois = keras.layers.Lambda(
                    lambda x: utils_graph.norm_boxes_graph(
                        x, keras.backend.shape(input_image)[1:3]))(caption_input_rois)
            else:
                mask_target_rois = mask_rpn_rois
                caption_target_rois = caption_rpn_rois

            # Generate detection targets
            # Sub-samples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            mask_rois, target_class_ids, mask_target_bbox, target_mask = \
                MaskDetectionTargetLayer(config, name='mask_proposal_targets')([
                    mask_target_rois, input_gt_class_ids, mask_gt_boxes, input_gt_masks])

            caption_rois, caption_target_bbox, target_caption, caption_target_scores = \
                CaptionDetectionTargetLayer(config, name='caption_proposal_targets')([
                    caption_target_rois, caption_gt_boxes, input_gt_captions, input_caption_gt_boxes_scores])

            mask_aligned_rois_bs = ROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                                            name="roi_align_mask_bs")(
                [mask_rois, input_image_meta] + macacnn_feature_maps)
            # Shape [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
            mask_aligned_rois = ROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],
                                         name='roi_align_mask')(
                [mask_rois, input_image_meta] + macacnn_feature_maps)
            # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
            caption_aligned_rois = ROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                                            name="roi_align_caption")(
                [caption_rois, input_image_meta] + macacnn_feature_maps)

            macacnn_class_logits, macacnn_class, mask_bbox = fpn_classifier_graph(mask_aligned_rois_bs,
                                                                                  config.POOL_SIZE,
                                                                                  config.NUM_CLASSES,
                                                                                  train_bn=config.TRAIN_BN)

            caption_bbox, caption_scores = build_fpn_bs_graph(caption_aligned_rois,
                                                              config.POOL_SIZE,
                                                              'caption',
                                                              train_bn=config.TRAIN_BN)

            macacnn_mask = build_fpn_mask_graph(mask_aligned_rois, config.NUM_CLASSES, train_bn=config.TRAIN_BN)

            macacnn_caption, _ = build_caption_layer_graph(caption_aligned_rois,
                                                           config.POOL_SIZE,
                                                           self.tokenizer,
                                                           mode,
                                                           self.config,
                                                           target_caption=target_caption)

            mask_output_rois = keras.layers.Lambda(lambda x: x * 1, name='mask_output_rois')(mask_rois)
            caption_output_rois = keras.layers.Lambda(lambda x: x * 1, name='caption_output_rois')(caption_rois)

            mask_rpn_class_loss = keras.layers.Lambda(lambda x: losses.rpn_class_loss_graph(*x),
                                                      name="mask_rpn_class_loss")(
                [input_mask_rpn_match, mask_rpn_class_logits])
            caption_rpn_class_loss = keras.layers.Lambda(lambda x: losses.rpn_class_loss_graph(*x),
                                                         name="caption_rpn_class_loss")(
                [input_caption_rpn_match, caption_rpn_class_logits])
            mask_rpn_bbox_loss = keras.layers.Lambda(lambda x: losses.rpn_bbox_loss_graph(config, *x),
                                                     name="mask_rpn_bbox_loss")(
                [input_mask_rpn_bbox, input_mask_rpn_match, mask_rpn_bbox])
            caption_rpn_bbox_loss = keras.layers.Lambda(lambda x: losses.rpn_bbox_loss_graph(config, *x),
                                                        name="caption_rpn_bbox_loss")(
                [input_caption_rpn_bbox, input_caption_rpn_match, caption_rpn_bbox])

            mask_class_loss = keras.layers.Lambda(lambda x: losses.mrcnn_class_loss_graph(*x),
                                                  name='macacnn_mask_class_loss')([
                target_class_ids, macacnn_class_logits, active_class_ids])
            mask_bbox_loss = keras.layers.Lambda(lambda x: losses.macacnn_mask_bbox_loss_graph(*x),
                                                 name="macacnn_mask_bbox_loss")(
                [mask_target_bbox, target_class_ids, mask_bbox])
            mask_loss = keras.layers.Lambda(lambda x: losses.mrcnn_mask_loss_graph(*x), name="macacnn_mask_loss")(
                [target_mask, target_class_ids, macacnn_mask])

            caption_bbox_loss = keras.layers.Lambda(lambda x: losses.mrcnn_bbox_loss_graph(*x),
                                                    name="macacnn_caption_bbox_loss")(
                [caption_target_bbox, caption_bbox])
            caption_bbox_score_loss = keras.layers.Lambda(lambda x: losses.macacnn_bbox_score_loss_graph(*x),
                                                          name='macarcnn_caption_bbox_score_loss')([
                caption_target_scores, caption_scores])
            caption_loss = keras.layers.Lambda(lambda x: losses.caption_loss_graph(*x), name='macacnn_caption_loss')(
                [target_caption, macacnn_caption])

            # Model
            inputs = [input_image, input_image_meta,
                      input_mask_rpn_match, input_mask_rpn_bbox, input_caption_rpn_match, input_caption_rpn_bbox,
                      input_gt_class_ids,
                      input_mask_gt_boxes, input_gt_masks, input_caption_gt_boxes, input_gt_captions,
                      input_caption_gt_boxes_scores]

            if not config.USE_RPN_ROIS:
                inputs.append(mask_input_rois)
                inputs.append(caption_input_rois)

            outputs = [mask_rpn_class_logits, caption_rpn_class_logits,
                       mask_rpn_class, mask_rpn_bbox, caption_rpn_class, caption_rpn_bbox,
                       caption_bbox, macacnn_mask, macacnn_caption, caption_scores,
                       mask_rpn_rois, caption_rpn_rois, mask_output_rois, caption_output_rois,
                       mask_rpn_class_loss, caption_rpn_class_loss,
                       mask_rpn_bbox_loss, caption_rpn_bbox_loss,
                       mask_class_loss,
                       caption_bbox_loss, mask_bbox_loss, caption_bbox_score_loss,
                       mask_loss, caption_loss]
            model = keras.Model(inputs, outputs, name='macacnn')
        else:
            # Shape [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
            mask_aligned_rois = ROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                                         name='roi_align_mask')(
                [mask_rpn_rois, input_image_meta] + macacnn_feature_maps)
            # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
            caption_aligned_rois = ROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                                            name="roi_align_caption")(
                [caption_rpn_rois, input_image_meta] + macacnn_feature_maps)

            macacnn_class_logits, macacnn_class, mask_bbox = fpn_classifier_graph(mask_aligned_rois,
                                                                                  config.POOL_SIZE,
                                                                                  config.NUM_CLASSES,
                                                                                  train_bn=config.TRAIN_BN)

            macacnn_caption_bbox, caption_scores = build_fpn_bs_graph(caption_aligned_rois,
                                                                      config.POOL_SIZE,
                                                                      'caption',
                                                                      train_bn=config.TRAIN_BN)

            mask_detections = MaskDetectionLayer(self.config, name='macacnn_mask_detection')([
                mask_rpn_rois, macacnn_class, mask_bbox, input_image_meta])
            mask_detection_boxes = keras.layers.Lambda(lambda x: x[..., :4])(mask_detections)

            caption_detections = CaptionDetectionLayer(self.config, name='macacnn_caption_detection')([
                caption_rpn_rois, caption_scores, macacnn_caption_bbox, input_image_meta])
            caption_detection_boxes = keras.layers.Lambda(lambda x: x[..., :4])(caption_detections)

            # Shape [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
            mask_aligned_detections = ROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],
                                               name='roi_detection_mask')(
                [mask_detection_boxes, input_image_meta] + macacnn_feature_maps)
            # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
            caption_aligned_detections = ROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                                                  name="roi_detection_caption")(
                [caption_detection_boxes, input_image_meta] + macacnn_feature_maps)

            macacnn_mask = build_fpn_mask_graph(mask_aligned_detections, config.NUM_CLASSES)

            macacnn_caption, attention_weights = build_caption_layer_graph(caption_aligned_detections,
                                                                           self.config.POOL_SIZE,
                                                                           self.tokenizer,
                                                                           mode,
                                                                           self.config)

            model = keras.Model([input_image, input_image_meta, input_anchors],
                                [caption_detections, macacnn_caption_bbox, macacnn_caption, attention_weights,
                                 macacnn_mask, mask_detections,
                                 mask_rpn_rois, caption_rpn_rois, mask_rpn_class, caption_rpn_class,
                                 mask_rpn_bbox, caption_rpn_bbox], name='macacnn')

        # Add multi-GPU support.
        if self.config.GPU_COUNT > 1:
            from scripts.parallel_model import ParallelModel
            model = ParallelModel(model, self.config.GPU_COUNT)
        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory name. Each directory corresponds to a model.
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("macacnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=True, skip_mismatch=True):
        """
        Load weights for the model.
        :param filepath:
        :param by_name:
        :param skip_mismatch:
        :return:
        """
        assert os.path.exists(filepath)

        log('Start loading weights from {}'.format(filepath))

        self.model.load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch)

        log('Load weights successfully.')

        # Update the log directory
        self.set_log_dir(filepath)
        log('Set log dir successfully.')

    def get_imagenet_weights(self, basic_name):
        """Downloads ImageNet trained weights
                Returns path to weights file.
                """
        if not basic_name:
            raise ValueError("basic_name should be a valid value rather {}".format(basic_name))

        import tensorflow.keras.utils.get_file as get_file
        if basic_name == 'resnet50':
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                     'releases/download/v0.2/' \
                                     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        elif basic_name == 'vgg16':
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                     'releases/download/v0.1/' \
                                     'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        elif basic_name == 'vgg19':
            TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                     'releases/download/v0.1/' \
                                     'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        return weights_path

    def compile(self, learning_rate, momentum=None):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.Nadam(lr=learning_rate,
                                           clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add losses and metrics
        loss_names = [
            "mask_rpn_class_loss", "caption_rpn_class_loss",
            "mask_rpn_bbox_loss", "caption_rpn_bbox_loss",
            "macacnn_mask_bbox_loss", "macacnn_caption_bbox_loss",
            'macacnn_mask_class_loss', 'macarcnn_caption_bbox_score_loss',
            "macacnn_mask_loss", "macacnn_caption_loss"
        ]
        for name in loss_names:
            # Add losses
            layer = self.model.get_layer(name)
            if name in self.exists_loss:
                continue
            if layer.output in self.model.losses:
                continue
            self.exists_loss.append(name)
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.model.add_loss(loss)
            log('insert ' + name + ' into model')
            # Add metrics
            if name in self.model.metrics_names:
                continue
            self.model.metrics_names.append(name)
            self.model.add_metric(loss, name=name, aggregation='mean')

        # Add L2 regularization
        # Skip gamma and beta weights of batch normalization layer
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
            for w in self.model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.model.add_loss(tf.add_n(reg_losses))
        log('insert L2 regularization into model')
        # Compile
        self.model.compile(optimizer=optimizer,
                           loss=[None] * len(self.model.outputs))

        # Add metrics for losses
        # for name in loss_names:
        #     if name in self.model.metrics_names:
        #         continue
        #     layer = self.model.get_layer(name)
        #     self.model.metrics_names.append(name)
        #     loss = (
        #             tf.reduce_mean(input_tensor=layer.output, keepdims=True)
        #             * self.config.LOSS_WEIGHTS.get(name, 1.))
        #     self.model.add_metric(loss, name=name, aggregation='mean')

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and model is not None:
            log("Selecting layers to train")

        model = model or self.model

        # In multi-GPU training, we wrap the model. Get layers
        # of the model because they have the weights.
        layers = model.inner_model.layers if hasattr(model, "inner_model") \
            else model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]macacnn\_[\w-]+(\d{4})\.hdf5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                log('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "macacnn_{}_*epoch*.hdf5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, verbose=0):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        # TODO: Update tge layer regex for MACA
        layer_regex_efficientnet = {
            # all layers but the backbone
            "heads": r"(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(block3.*)|(block4.*)|(block5.*)|(block6.*)|(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(block4.*)|(block5.*)|(block6.*)|(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(block5.*)|(block6.*)|(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }

        layer_regex_resnet = {
            # all layers but the backbone
            "heads": r"(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(macacnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }

        layer_regex_vggnet = {
            "all": ".*"
        }

        if self.config.BACKBONE.startswith('efficientnet'):
            layer_regex = layer_regex_efficientnet
        elif self.config.BACKBONE.startswith('resnet'):
            layer_regex = layer_regex_resnet
        else:
            layer_regex = layer_regex_vggnet

        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        log('Start building datasets...')
        train_generator = DataGenerator(train_dataset, self.config, shuffle=True,
                                        augmentation=augmentation)
        log('Successfully build train dataset')
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True)
        log('Successfully build val dataset')

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            # keras.callbacks.TensorBoard(log_dir=self.log_dir,
            #                             histogram_freq=0, write_graph=False, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True,
                                            save_best_only=False),
        ]
        # Add customer callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks
        log('Successfully build callbacks')

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        log('Successfully compile, now start training...')

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        # if os.name is 'nt':
        #     workers = 0
        # else:
        #     import multiprocessing
        #     workers = multiprocessing.cpu_count()
        #
        # log('Worker nums: {}'.format(workers))

        self.model.fit(train_generator,
                       initial_epoch=self.epoch,
                       epochs=epochs,
                       steps_per_epoch=self.config.STEPS_PER_EPOCH,
                       callbacks=callbacks,
                       validation_data=val_generator,
                       validation_steps=self.config.VALIDATION_STEPS,
                       max_queue_size=100,
                       #  workers=workers,
                       #  use_multiprocessing=workers > 1,
                       verbose=verbose)
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = utils.mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = utils_graph.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES]))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
            # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, caption_detections, mask_detections, maca_masks, maca_captions, attentions,
                          original_image_shape, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, score)] in normalized coordinates
        maca_mask: [N, height, width, ]
        maca_caption: [N, MAX_LENGTH, VOCAB_SIZE]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        mask_rois: [N, 4]

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first idx == 0.
        zero_ix = np.where(caption_detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else caption_detections.shape[0]

        # Extract boxes, scores, and class-specific masks
        caption_boxes = caption_detections[:N, :4]
        caption_scores = caption_detections[:N, 4]
        captions = maca_captions[:N, :, :]
        attentions = attentions[:N, :, :]

        mask_zero_ix = np.where(mask_detections[:, 4] == 0)[0]
        N = mask_zero_ix[0] if mask_zero_ix.shape[0] > 0 else mask_detections.shape[0]
        mask_boxes = mask_detections[:N, :4]
        mask_class_ids = mask_detections[:N, 4].astype(np.int32)
        mask_scores = mask_detections[:N, 5]
        masks = maca_masks[np.arange(N), :, :, mask_class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        caption_boxes = np.divide(caption_boxes - shift, scale)
        mask_boxes = np.divide(mask_boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        caption_boxes = utils.denorm_boxes(caption_boxes, original_image_shape[:2])
        mask_boxes = utils.denorm_boxes(mask_boxes, original_image_shape[:2])

        if caption_boxes.shape[0] >= 2:
            ix = utils.non_max_suppression(caption_boxes, caption_scores, 0.2)
            caption_boxes = caption_boxes[ix]
            caption_scores = caption_scores[ix]
            captions = captions[ix]
            attentions = attentions[ix]

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where((caption_boxes[:, 2] - caption_boxes[:, 0]) *
                              (caption_boxes[:, 3] - caption_boxes[:, 1]) <= 50)[0]
        if exclude_ix.shape[0] > 0:
            caption_boxes = np.delete(caption_boxes, exclude_ix, axis=0)
            caption_scores = np.delete(caption_scores, exclude_ix, axis=0)
            captions = np.delete(captions, exclude_ix, axis=0)
            attentions = np.delete(attentions, exclude_ix, axis=0)

        # ix = utils.non_max_suppression(mask_boxes, mask_scores, 0.3)
        # masks = masks[:, :, ix]
        # mask_scores = mask_scores[ix]

        # Filter out mask_rois with zero area.
        mask_exclude_ix = np.where(
            (mask_boxes[:, 2] - mask_boxes[:, 0]) * (mask_boxes[:, 3] - mask_boxes[:, 1]) <= 0)[0]
        if mask_exclude_ix.shape[0] > 0:
            masks = np.delete(masks, mask_exclude_ix, axis=2)
            mask_boxes = np.delete(mask_boxes, mask_exclude_ix, axis=0)
            mask_class_ids = np.delete(mask_class_ids, mask_exclude_ix, axis=0)
            mask_scores = np.delete(mask_scores, mask_exclude_ix, axis=0)

        N = masks.shape[0]
        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], mask_boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        # Extract captions
        captions_sequences = np.argmax(captions, axis=2)
        captions = self.tokenizer.sequences_to_texts(captions_sequences)
        for i, caption in enumerate(captions):
            captions[i] = caption.split('<end>')[0]
            captions[i] = captions[i].replace('<unk>', '')

        return caption_boxes, caption_scores, full_masks, captions, mask_class_ids, attentions

    def detect(self, images, verbose=0, track=False):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because keras requires it
        # TODO: Can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)

        outputs = self.model.predict([molded_images, image_metas, anchors], verbose=verbose)

        masks = outputs[4]
        mask_detections = outputs[5]
        detections = outputs[0]
        captions = outputs[2]
        attention_weights = outputs[3]

        # process results
        results = []
        for i, image in enumerate(images):
            final_rois, final_scores, final_masks, final_captions, class_ids, attentions = \
                self.unmold_detections(detections[i], mask_detections[i], masks[i], captions[i], attention_weights[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])

            if not track:
                results.append({
                    'rois': final_rois,
                    'scores': final_scores,
                    'masks': final_masks,
                    'captions': final_captions,
                    'attentions': attentions,
                    "class_ids": class_ids,
                })
            else:
                if self.pre_frame_detection is None:
                    results.append({
                        'rois': final_rois,
                        'scores': final_scores,
                        'masks': final_masks,
                        'captions': final_captions,
                        "class_ids": class_ids,
                    })
                    self.pre_frame_detection = results[0]
                else:
                    track_final_rois, track_final_scores, track_final_captions = \
                        utils.compute_track_result(self.pre_frame_detection, final_rois, final_scores,
                                                   final_captions, threshold=0.7)
                    results.append({
                        'rois': track_final_rois,
                        'scores': track_final_scores,
                        'masks': final_masks,
                        'captions': track_final_captions,
                        "class_ids": class_ids,
                    })
                    self.pre_frame_detection = results[0]

        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):

        assert self.mode == 'inference', 'Create model in inference mode.'
        assert len(molded_images) == self.config.BATCH_SIZE, \
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, 'Image must have the same size.'

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it.
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("iamge_metas", image_metas)
            log("anchors", anchors)

        # Run object detection
        detections, _, captions, attention_weights, masks, _, _, _, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=verbose)

        # process results
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_scores, final_masks, final_captions = \
                self.unmold_detections(detections[i], masks[i], captions[i],
                                       image.shape, molded_images[i].shape,
                                       window)

            results.append({
                'rois': final_rois,
                'scores': final_scores,
                'masks': final_masks,
                'captions': final_captions,
            })

        return results

    def get_anchors(self, image_shape):
        """Returns anchors generated from the feature map."""
        backbone_shapes = utils.compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                               self.config.RPN_ANCHOR_RATIOS,
                                               backbone_shapes,
                                               self.config.BACKBONE_STRIDES,
                                               self.config.RPN_ANCHOR_STRIDE)

            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # Normalize coordinates
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.model.layers:
            # if layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        # if model.uses_learning_phase and not isinstance(keras.backend.learning_phase(), int):
        #     inputs += [keras.backend.learning_phase()]
        kf = keras.backend.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self._get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        # if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #     model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            utils.log(k, v)
        return outputs_np


if __name__ == '__main__':
    from config import Config
    import json

    mode = 'training'
    Config.NAME = 'cocovg'
    Config.IMAGES_PER_GPU = 1
    Config.NUM_CLASSES = 80 + 1

    config = Config()
    # config.display()

    with open("/Users/liyiming/Desktop/Birmingham Life/project/DATASET/COCOVG/tokenizer.json", 'r') as load_f:
        js_tok = json.load(load_f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(js_tok)

    model = MACA(mode, config, "../", tokenizer)
    model.model.summary()
    # model.model.summary()
    # model.model.save('/Users/liyiming/Desktop/Birmingham Life/project/yxl1215/code/version1/Mask-Dence-Cap-R-CNN/macacnn_improvement')
    # tf.keras.utils.plot_model(model.model, to_file=os.path.join('../', 'model.png'), show_shapes=True,
    #                           show_layer_names=True)

    # image = np.random.random([1024, 1024, 3])
    # image = np.abs(image)
    # molded_images, image_metas, windows = model.mold_inputs([image])
    # anchors = model.get_anchors(image.shape)
    # anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
    # log("image", image)
    # log("molded_images", molded_images)
    # log("image_metas", image_metas)
    # log("anchors", anchors)
    # outputs = model.model.predict([molded_images, image_metas, anchors], verbose=0)
