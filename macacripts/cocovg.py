from config import Config
from Dataset import Dataset

from pycocotools import mask as maskUtils

import os
import json
import numpy as np


class CocoVGConfig(Config):
    """Configuration for training on combined COCOVG.
    Derives from the base Config class and overrides values specific
    to the COCOVG dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cocovg"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class CocoVGDataset(Dataset):

    def initalize_dataset(self, root_dir, type):
        assert type in ['train', 'val']
        image_dir = os.path.join(root_dir, type)

        with open(os.path.join(root_dir, '{}_ann.json'.format(type)), 'r') as load_f:
            annotations = json.load(load_f)

        for id in Config.class_ids.keys():
            self.add_class('cocovg', id, Config.class_ids[id])

        for ann in annotations:
            self.add_image('cocovg', image_id=ann['image_id'],
                           path=os.path.join(image_dir, '{}.jpg'.format(ann['image_id'])),
                           width=ann['image_width'],
                           height=ann['image_height'],
                           annotations=ann)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info['source'] != 'cocovg':
            return super(CocoVGDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = image_info['annotations']
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        coco_ann = annotations['coco_ann']
        for ann in coco_ann:
            class_id = self.map_source_class_id("cocovg.{}".format(ann['category_id']))
            if class_id:
                m = self.annToMask(ann, image_info['height'], image_info['width'])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
            instance_masks.append(m)
            class_ids.append(class_id)

        # Pcak instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(CocoVGDataset, self).load_mask(image_id)

    def load_captions(self, image_id):
        image_info = self.image_info[image_id]
        instance_captions = []
        annotations = image_info['annotations']
        vg_ann = annotations['vg_ann']
        for ann in vg_ann:
            caption = ann['tokenized_padded_phrase']
            instance_captions.append(caption)

        captions = np.stack(instance_captions, axis=0)
        return captions

    def load_caption_boxes(self, image_id):
        image_info = self.image_info[image_id]
        instance_bboxes = []
        annotations = image_info['annotations']
        vg_ann = annotations['vg_ann']
        for ann in vg_ann:
            x1 = ann['x']
            y1 = ann['y']
            x2 = x1 + ann['width']
            y2 = y1 + ann['height']
            instance_bboxes.append([y1, x1, y2, x2])
        bboxes = np.stack(instance_bboxes, axis=0)
        scores = np.ones(bboxes.shape[0])
        return bboxes, scores

    def annToMask(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        m = maskUtils.decode(rle)
        return m


if __name__ == '__main__':
    dataset = CocoVGDataset()
    ROOT_DIR = '/Users/liyiming/Desktop/Birmingham Life/project/DATASET/COCOVG'
    dataset.initalize_dataset(ROOT_DIR, 'train')
    dataset.prepare()









