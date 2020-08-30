"""Data Generator."""

import numpy as np
from tensorflow import keras

from macacripts.utils import compute_backbone_shapes
from macacripts import utils_graph, utils


def load_image_gt(dataset, config, image_id, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes, captions).
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    captions: [instance_count, MAX_LEN_CAPTIONS]
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    caption_boxes, caption_box_scores = dataset.load_caption_boxes(image_id)
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)
    caption_boxes = utils.resize_boxes(caption_boxes, scale, padding, crop)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    mask_bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if config.USE_MINI_MASK:
        mask = utils.minimize_mask(mask_bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = utils_graph.compose_image_meta(image_id, original_shape, image.shape,
                                                window, scale, active_class_ids)

    captions = dataset.load_captions(image_id)

    return image, image_meta, class_ids, mask_bbox, mask, captions, caption_boxes, caption_box_scores


def build_rpn_targets(image_shape, anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    # no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


class DataGenerator(keras.utils.Sequence):
    """An iterable that returns images and corresponding target class ids,
    bounding box deltas, and masks. It inherits from keras.utils.Sequence to avoid data redundancy
    when multiprocessing=True.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python iterable. Upon calling __getitem__() on it, the
    iterable returns two lists, inputs and outputs. The contents
    of the lists differ depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_mask_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.
    - gt_captions: [batch, MAX_GT_INSTANCES, MAX_CAPTIONS_LENGTH]
    - gt_caption_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """

    def __init__(self, dataset, config, shuffle=True, augmentation=None,
                 random_rois=0, detection_targets=False):
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      self.backbone_shapes,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)

        self.shuffle = shuffle
        self.augmentation = augmentation
        self.random_rois = random_rois
        self.batch_size = self.config.BATCH_SIZE
        self.detection_targets = detection_targets

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        b = 0
        image_index = -1
        while b < self.batch_size:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(self.image_ids)

            if self.shuffle and image_index == 0:
                np.random.shuffle(self.image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = self.image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes_mask, gt_masks, gt_captions, gt_boxes_caption, \
                    caption_box_scores = load_image_gt(self.dataset, self.config, image_id,
                                                                         augmentation=self.augmentation)

            if not np.any(gt_class_ids > 0):
                continue

            # Mask RPN Targets
            mask_rpn_match, mask_rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                              gt_boxes_mask, self.config)

            # Caption RPN Targets
            caption_rpn_match, caption_rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                                    gt_boxes_caption, self.config)

            # Mask R-CNN Targets
            # if self.random_rois:
            #     rpn_rois = generate_random_rois(
            #         image.shape, self.random_rois, gt_class_ids, gt_boxes_mask)
            #     if self.detection_targets:
            #         rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
            #             build_detection_targets(
            #                 rpn_rois, gt_class_ids, gt_boxes_mask, gt_masks, self.config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_mask_rpn_match = np.zeros(
                    [self.batch_size, self.anchors.shape[0], 1], dtype=mask_rpn_match.dtype)
                batch_mask_rpn_bbox = np.zeros(
                    [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=mask_rpn_bbox.dtype)

                batch_caption_rpn_match = np.zeros(
                    [self.batch_size, self.anchors.shape[0], 1], dtype=caption_rpn_match.dtype)
                batch_caption_rpn_bbox = np.zeros(
                    [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=caption_rpn_bbox.dtype)

                batch_images = np.zeros(
                    (self.batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)

                batch_mask_gt_boxes = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

                batch_gt_captions = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES, gt_captions.shape[1]), dtype=np.int32)
                batch_caption_gt_boxes = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_caption_gt_boxes_scores = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.float32)

                # if self.random_rois:
                #     batch_rpn_rois = np.zeros(
                #         (self.batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                #     if self.detection_targets:
                #         batch_rois = np.zeros(
                #             (self.batch_size,) + rois.shape, dtype=rois.dtype)
                #         batch_mrcnn_class_ids = np.zeros(
                #             (self.batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                #         batch_mrcnn_bbox = np.zeros(
                #             (self.batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                #         batch_mrcnn_mask = np.zeros(
                #             (self.batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes_mask.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes_mask.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes_mask = gt_boxes_mask[ids]
                gt_masks = gt_masks[:, :, ids]

            if gt_boxes_caption.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes_mask.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_captions = gt_captions[ids]
                gt_boxes_caption = gt_boxes_caption[ids]
                caption_box_scores = caption_box_scores[ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_mask_rpn_match[b] = mask_rpn_match[:, np.newaxis]
            batch_mask_rpn_bbox[b] = mask_rpn_bbox

            batch_caption_rpn_match[b] = caption_rpn_match[:, np.newaxis]
            batch_caption_rpn_bbox[b] = caption_rpn_bbox

            batch_images[b] = utils.mold_image(image.astype(np.float32), self.config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids

            batch_mask_gt_boxes[b, :gt_boxes_mask.shape[0]] = gt_boxes_mask
            batch_caption_gt_boxes[b, :gt_boxes_caption.shape[0]] = gt_boxes_caption
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_captions[b, :gt_captions.shape[0]] = gt_captions
            batch_caption_gt_boxes_scores[b, :caption_box_scores.shape[0]] = caption_box_scores
            # if self.random_rois:
            #     batch_rpn_rois[b] = rpn_rois
            #     if self.detection_targets:
            #         batch_rois[b] = rois
            #         batch_mrcnn_class_ids[b] = mrcnn_class_ids
            #         batch_mrcnn_bbox[b] = mrcnn_bbox
            #         batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

        inputs = [batch_images, batch_image_meta,
                  batch_mask_rpn_match, batch_mask_rpn_bbox, batch_caption_rpn_match, batch_caption_rpn_bbox,
                  batch_gt_class_ids,
                  batch_mask_gt_boxes, batch_gt_masks, batch_caption_gt_boxes, batch_gt_captions,
                  batch_caption_gt_boxes_scores]
        outputs = []

        # if self.random_rois:
        #     inputs.extend([batch_rpn_rois])
        #     if self.detection_targets:
        #         inputs.extend([batch_rois])
        #         # Keras requires that output and targets have the same number of dimensions
        #         batch_mrcnn_class_ids = np.expand_dims(
        #             batch_mrcnn_class_ids, -1)
        #         outputs.extend(
        #             [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        return inputs, outputs


if __name__ == '__main__':
    from scripts.cocovg import CocoVGDataset, CocoVGConfig

    config = CocoVGConfig()
    dataset = CocoVGDataset()
    dataset.initalize_dataset('/Users/liyiming/Desktop/Birmingham Life/project/DATASET/COCOVG', 'train')
    dataset.prepare()

    dg = DataGenerator(dataset, config, shuffle=True)
