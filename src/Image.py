import os
from pyvips import Image as VipsImage
import cv2
import numpy as np
import warnings

class Image(object):
   def __init__(self, path, distance):
      self.path = path
      self.filename = os.path.basename(path)
      self.distance = distance
      self.annotations = []
      self.scale = 1

   def add_annotation(self, annotation):
      self.annotations.append(annotation)

   def set_target_distance(self, distance):
      self.scale = self.distance / distance
      for annotation in self.annotations:
         annotation.set_scale(self.scale)

   def generate_train_patches(self, images_path, masks_path, dimension):
      image = VipsImage.new_from_file(self.path).resize(self.scale)

      if image.width < dimension or image.height < dimension:
         warnings.warn('Image "{}" is smaller than the crop dimension!'.format(self.filename))

      masks = []
      for annotation in self.annotations:
         mask = np.zeros((image.height, image.width), dtype=np.uint8)
         cv2.circle(mask, annotation.get_center(), annotation.get_radius(), 1, -1)
         masks.append(mask.astype(np.bool))

      image_paths = []
      mask_paths = []
      mean_pixels = []

      for i, annotation in enumerate(self.annotations):
         image_file = '{}_{}.jpg'.format(self.filename, i)
         image_crop, mask_crops = self.generate_annotation_crop(image, masks, annotation, dimension)
         mask_file = self.save_mask(mask_crops, image_file, masks_path)
         image_crop.write_to_file(os.path.join(images_path, image_file))
         image_paths.append(image_file)
         mask_paths.append(mask_file)
         np_crop = np.ndarray(buffer=image_crop.write_to_memory(), shape=[image_crop.height, image_crop.width, image_crop.bands], dtype=np.uint8)
         mean_pixels.append(np_crop.reshape((-1, 3)).mean(axis = 0))

      return image_paths, mask_paths, mean_pixels

   def generate_annotation_crop(self, image, masks, annotation, dimension):
      width, height = image.width, image.height

      crop_width = min(width, dimension)
      crop_height = min(height, dimension)
      current_crop_dimension = np.array([crop_width, crop_height])

      center = np.array(annotation.get_center())
      topLeft = np.round(center - current_crop_dimension / 2).astype(np.int32)
      bottomRight = np.round(center + current_crop_dimension / 2).astype(np.int32)
      offset = [0, 0]
      if topLeft[0] < 0: offset[0] = abs(topLeft[0])
      if topLeft[1] < 0: offset[1] = abs(topLeft[1])
      if bottomRight[0] > width: offset[0] = width - bottomRight[0]
      if bottomRight[1] > height: offset[1] = height - bottomRight[1]

      topLeft += offset
      bottomRight += offset

      image_crop = image.extract_area(topLeft[0], topLeft[1], current_crop_dimension[0], current_crop_dimension[1])
      mask_crops = [mask[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]] for mask in masks]

      return image_crop, mask_crops

   def save_mask(self, masks, filename, path):
      mask_store = [mask for mask in masks if np.any(mask)]
      mask_file = '{}.npz'.format(filename)
      np.savez_compressed(os.path.join(path, mask_file), masks=mask_store)

      return mask_file

   def generate_random_crop(self, target_path, dimension, prefix=''):
      image = VipsImage.new_from_file(self.path)
      x = np.random.randint(image.width - dimension)
      y = np.random.randint(image.height - dimension)
      image_crop = image.extract_area(x, y, dimension, dimension)
      filename = self.filename
      if prefix != '':
         filename = '{}-{}'.format(prefix, filename)
      image_crop.write_to_file(os.path.join(target_path, filename))

      return filename
