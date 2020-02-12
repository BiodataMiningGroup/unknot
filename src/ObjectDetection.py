import numpy as np
import os.path
import imgaug.augmenters as iaa
import json
from pyvips import Image as VipsImage

from . import PatchesCollection
from . import Dataset as ImageDataset
from . import utils

from .mrcnn import config as mrcnn_config
from .mrcnn import utils as mrcnn_utils
from .mrcnn import model as mrcnn_model

class Config(mrcnn_config.Config):
   def __init__(self, train_patches, config={}):
      self.NAME = 'unknot'
      # Add one for the background class (0).
      self.NUM_CLASSES = 2
      # Disable validation since we do not have ground truth.
      self.VALIDATION_STEPS = 0
      self.MEAN_PIXEL = np.array(train_patches.mean_pixel)

      self.AUGMENTATION = iaa.SomeOf((0, None), [
         iaa.Fliplr(1.0),
         iaa.Flipud(1.0),
         iaa.Affine(rotate=[90, 180, 270]),
         iaa.GaussianBlur(sigma=(1.0, 2.0)),
         iaa.JpegCompression(compression=(25, 50)),
      ], random_order=True)

      for key, value in config.items():
         setattr(self, key, value)

      super().__init__()

class TrainingConfig(Config):
   def __init__(self, train_patches, config={}):
      self.IMAGE_MAX_DIM = train_patches.crop_dimension
      super().__init__(train_patches, config)
      # In total, we want to train with about 2000 images per epoch.
      self.STEPS_PER_EPOCH = round(2000 / self.IMAGES_PER_GPU)


class InferenceConfig(Config):
   def __init__(self, train_patches, config={}):
      self.IMAGES_PER_GPU = 1
      self.IMAGE_MIN_DIM = 64
      self.IMAGE_RESIZE_MODE = "pad64"
      super().__init__(train_patches, config)

class Dataset(mrcnn_utils.Dataset):
   def __init__(self, images, name='no_name', masks=[], classes={}, ignore_classes=[]):
      super().__init__()
      # Convert to the required dict with image IDs.
      images = {k: v for k, v in enumerate(images)}

      self.images = images
      self.masks = masks
      self.name = name
      self.classes = classes
      # Always ignore the background class.
      self.ignore_classes = set([0] + ignore_classes)

   def prepare(self):
      for class_id, class_name in self.classes.items():
         self.add_class(self.name, class_id, class_name)

      for image_id, image_file in self.images.items():
         self.add_image(self.name, image_id, image_file)

      super().prepare()

   def load_mask(self, image_index):
      file = self.masks[image_index]
      data = np.load(file, allow_pickle=True)
      classes = []
      masks = []

      for mask in data['masks']:
         source_class_id = 1
         if source_class_id not in self.ignore_classes:
             classes.append(self.map_source_class_id('{}.{}'.format(self.name, source_class_id)))
             masks.append(mask)

      if len(classes) == 0:
         return super().load_mask(image_index)

      classes = np.array(classes, dtype=np.int32)
      masks = np.stack(masks, axis = 2).astype(np.bool)

      return masks, classes

class TrainingDataset(Dataset):
   def __init__(self, train_patches):
      images = train_patches.get_images_paths()
      masks = train_patches.get_masks_paths()
      classes = {1: 'Interesting'}
      super().__init__(images=images, masks=masks, classes=classes)

class InferenceDataset(Dataset):
   def __init__(self, images):
      classes = {1: 'Interesting'}
      super().__init__(images=images, classes=classes)

class ObjectDetector(object):
   def __init__(self, model_dir):
      self.model_dir = model_dir

   def perform_training(self, annotation_patches, scheme, config={}, initial_model=None):
      if not isinstance(annotation_patches, PatchesCollection.PatchesCollection):
         raise TypeError('The annotation patches must be a PatchesCollection.')

      if not annotation_patches.exists:
         raise RuntimeError('The annotation patches do not exist.')

      utils.ensure_dir(self.model_dir)
      train_config = TrainingConfig(annotation_patches, config)
      train_dataset = TrainingDataset(annotation_patches)

      train_config.display()
      train_dataset.prepare()
      model = mrcnn_model.MaskRCNN(mode="training", config=train_config, model_dir=self.model_dir)

      if initial_model:
         exclude_layers = [
            "mrcnn_class_logits",
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask",
         ]
         model.load_weights(initial_model, by_name=True, exclude=exclude_layers)

      epochs = 0
      for train_step in scheme:
         print('Train step: ', train_step)
         epochs += train_step['epochs']
         model.train(train_dataset,
            val_dataset=None,
            learning_rate=train_step['learning_rate'],
            epochs=epochs,
            layers=train_step['layers'],
            augmentation=train_config.AUGMENTATION
         )

      model_path = os.path.join(self.model_dir, "mask_rcnn_final.h5")
      model.keras_model.save_weights(model_path)

   def perform_inference(self, annotation_patches, dataset, target_dir):
      if not isinstance(dataset, ImageDataset.Dataset):
         raise TypeError('The dataset must be a Dataset.')

      images = [image.path for image in dataset.get_test_images()]
      config = InferenceConfig(annotation_patches)
      dataset = InferenceDataset(images)

      config.display()
      dataset.prepare()

      utils.ensure_dir(target_dir)

      model_path = os.path.join(self.model_dir, "mask_rcnn_final.h5")
      if not os.path.exists(model_path):
         raise RuntimeError('The trained model file does not exist. Perform training first.')

      model = mrcnn_model.MaskRCNN(mode="inference", config=config, model_dir=self.model_dir)
      model.load_weights(model_path, by_name=True)

      for i, image_info in enumerate(dataset.image_info):
         print('Processing image {}'.format(os.path.basename(image_info['path'])))
         image = dataset.load_image(i)
         results = model.detect([image])
         self.process_inference_result(results[0], image_info, target_dir)

   def process_inference_result(self, result, image_info, target_dir):
      filename = os.path.basename(image_info['path'])
      points = []
      for roi, score in zip(result['rois'], result['scores']):
         # ROIs are stored as (y1, x1, y2, x2).
         y = min(roi[0], roi[2])
         x = min(roi[1], roi[3])
         h = abs(roi[0] - roi[2])
         w = abs(roi[1] - roi[3])
         rx = round(w / 2)
         ry = round(h / 2)
         r = max(rx, ry)
         points.append([int(x + rx), int(y + ry), int(r), float(score)])

      path = os.path.join(target_dir, '{}.json'.format(filename))
      with open(path, 'w') as outfile:
         json.dump(points, outfile)

      image = VipsImage.new_from_file(image_info['path'])
      width, height = image.width, image.height

      mask = np.zeros((height, width), dtype=np.bool)
      for m in result['masks']:
          mask += m
      mask = mask.astype(np.uint8) * 255
      path = os.path.join(target_dir, '{}.png'.format(filename))
      image = VipsImage.new_from_memory(mask, width, height, 1, 'uchar')
      image.write_to_file(path)

