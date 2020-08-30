"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "efficientnet-b3"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 512

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "mask_rpn_class_loss": 1.,
        "caption_rpn_class_loss": 1.,
        "mask_rpn_bbox_loss": 1.,
        "caption_rpn_bbox_loss": 1.,
        "macacnn_mask_bbox_loss": 1.,
        "macacnn_caption_bbox_loss": 1.,
        "macacnn_mask_class_loss": 1.5,
        "macarcnn_caption_bbox_score_loss": 1.,
        "macacnn_mask_loss": 1.5,
        "macacnn_caption_loss": 1.5,
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    # Caption Part Paramters

    TOKENIZER = None

    EMBEDDING_DIM = 256
    UNITS = 512
    VOCAB_SIZE = 5000 + 1  # 5000 contains the dictionary and <start> <end> <unk> <pad> tokens
    MAX_LENGTH = 12

    class_ids = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
        9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
        24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
        47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
        55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
        63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
        74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
        81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush',
    }

    MASK_COLORS = [(0.0, 1.0, 0.07407407407407396), (0.0, 0.44444444444444464, 1.0), (0.44444444444444464, 1.0, 0.0),
              (0.0, 0.44444444444444464, 1.0), (0.5925925925925926, 0.0, 1.0), (0.44444444444444464, 1.0, 0.0),
              (0.44444444444444464, 1.0, 0.0), (0.44444444444444464, 1.0, 0.0), (0.666666666666667, 0.0, 1.0),
              (1.0, 0.0, 0.37037037037037024), (1.0, 0.6666666666666666, 0.0), (0.0, 1.0, 0.2962962962962963),
              (1.0, 0.2962962962962963, 0.0), (0.0, 0.8888888888888893, 1.0), (1.0, 0.0, 0.9629629629629637),
              (1.0, 0.0, 0.14814814814814792), (1.0, 0.0, 0.518518518518519), (0.37037037037037046, 1.0, 0.0),
              (0.7407407407407405, 0.0, 1.0), (0.14814814814814792, 0.0, 1.0), (0.0740740740740744, 0.0, 1.0),
              (0.44444444444444464, 0.0, 1.0), (1.0, 0.37037037037037035, 0.0), (0.962962962962963, 1.0, 0.0),
              (1.0, 0.0, 0.7407407407407405), (1.0, 0.9629629629629629, 0.0), (1.0, 0.0, 0.22222222222222232),
              (0.0, 1.0, 0.5925925925925926), (0.0, 1.0, 0.5185185185185182), (0.2962962962962963, 1.0, 0.0),
              (0.07407407407407418, 1.0, 0.0), (0.0, 1.0, 0.9629629629629628), (1.0, 0.14814814814814814, 0.0),
              (0.8148148148148149, 1.0, 0.0), (0.0, 0.9629629629629628, 1.0), (0.22222222222222232, 0.0, 1.0),
              (0.5925925925925926, 1.0, 0.0), (0.5185185185185186, 1.0, 0.0), (0.0, 0.5925925925925926, 1.0),
              (1.0, 0.0, 0.8148148148148149), (0.0, 1.0, 0.7407407407407405), (0.6666666666666667, 1.0, 0.0),
              (1.0, 0.0, 0.8888888888888893), (1.0, 0.8148148148148148, 0.0), (0.0, 0.6666666666666665, 1.0),
              (1.0, 0.4444444444444444, 0.0), (1.0, 0.5925925925925926, 0.0), (0.0, 1.0, 0.8148148148148149),
              (0.8888888888888888, 1.0, 0.0), (1.0, 0.0, 0.29629629629629584), (1.0, 0.0, 0.0740740740740744),
              (0.0, 0.2962962962962967, 1.0), (0.0, 0.8148148148148149, 1.0), (0.0, 1.0, 0.8888888888888888),
              (1.0, 0.0, 0.5925925925925926), (1.0, 0.0, 0.44444444444444464), (0.0, 0.14814814814814836, 1.0),
              (0.0, 1.0, 0.37037037037037024), (0.22222222222222232, 1.0, 0.0), (1.0, 0.7407407407407407, 0.0),
              (0.9629629629629628, 0.0, 1.0), (1.0, 0.0, 0.666666666666667), (0.8148148148148149, 0.0, 1.0),
              (0.0, 0.0, 1.0), (0.7407407407407409, 1.0, 0.0), (0.0, 1.0, 0.22222222222222232),
              (0.14814814814814836, 1.0, 0.0), (1.0, 0.07407407407407407, 0.0), (0.37037037037037024, 0.0, 1.0),
              (0.0, 0.37037037037037024, 1.0), (0.0, 1.0, 0.6666666666666665), (0.29629629629629584, 0.0, 1.0),
              (0.0, 1.0, 0.14814814814814792), (1.0, 0.2222222222222222, 0.0), (1.0, 0.0, 0.0),
              (0.0, 0.5185185185185182, 1.0), (0.0, 1.0, 0.4444444444444442), (0.0, 1.0, 0.0),
              (0.518518518518519, 0.0, 1.0), (1.0, 0.5185185185185185, 0.0), (0.0, 0.7407407407407409, 1.0)]

    CAPTION_COLORS = [(44, 172, 172), (172, 44, 172), (172, 172, 44), (172, 75, 45), (44, 172, 44)]

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        # for a in dir(self):
        #     if not a.startswith("__") and not callable(getattr(self, a)):
        #         print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
