import json
import csv
import numpy as np
from utils import normalize_path, ensure_dir
import os
from CircleAnnotation import CircleAnnotation
from Image import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import shutil

class Dataset(object):

   def __init__(self, config_path):
      with open(config_path, 'r') as f:
         self.config = json.load(f)
      self.config_dir = os.path.dirname(config_path)
      self.name = self.config['name']
      self.crop_dimension = self.config.get('crop_dimension', 512)

   def get_config_path(self, key):
      return normalize_path(self.config_dir, self.config[key])

   def get_metadata(self):
      metadata_path = self.get_config_path('metadata')
      with open(metadata_path, 'r') as file:
         reader = csv.reader(file)
         next(reader)
         metadata = {row[0]: float(row[1]) for row in reader}

      return metadata

   def get_mean_distance(self):
      metadata = self.get_metadata()

      return np.fromiter(metadata.values(), dtype=float).mean()

   def read_report(self, report_path):
      metadata = self.get_metadata()
      images_dir = self.get_config_path('images_dir')
      images = {}
      with open(report_path, 'r') as file:
         reader = csv.reader(file)
         next(reader)
         for row in reader:
            image_filename = row[8]
            annotation = CircleAnnotation(row)
            if image_filename not in images:
               image_path = os.path.join(images_dir, image_filename)
               image_distance = metadata[image_filename]
               images[image_filename] = Image(image_path, image_distance)
            images[image_filename].add_annotation(annotation)

      return images.values()

   def read_train_report(self):
      train_report = self.get_config_path('train_annotations_report')

      return self.read_report(train_report)

   def read_test_report(self):
      test_report = self.get_config_path('test_annotations_report')

      return self.read_report(test_report)

   def generate_scale_transfer(self, target_dataset):
      if not isinstance(target_dataset, Dataset):
         raise TypeError('The target dataset must be a Dataset.')

      target_path = self.get_config_path('scale_transfer_target_path')
      train_json_path = os.path.join(target_path, 'train.json')

      if os.path.isfile(train_json_path):
         with open(train_json_path, 'r') as f:
            train_json = json.load(f)
         if train_json['for_dataset'] == target_dataset.name and train_json['crop_dimension'] == self.crop_dimension:
            return train_json

      images_target_path = os.path.join(target_path, 'images')
      ensure_dir(images_target_path)
      masks_target_path = os.path.join(target_path, 'masks')
      ensure_dir(masks_target_path)
      target_distance = target_dataset.get_mean_distance()
      images = self.read_train_report()

      executor = ThreadPoolExecutor(max_workers=self.config['workers'])
      jobs = []
      for image in images:
         image.set_target_distance(target_distance)
         jobs.append(executor.submit(image.generate_train_patches, images_target_path, masks_target_path, self.crop_dimension))

      images = []
      masks = []
      mean_pixels = []

      for job in as_completed(jobs):
         i, m, p = job.result()
         if i is not False:
            images.extend(i)
            masks.extend(m)
            mean_pixels.extend(p)

      mean_pixel = np.array(mean_pixels).mean(axis = 0).tolist()

      train_json = {
         'for_dataset': target_dataset.name,
         'crop_dimension': self.crop_dimension,
         'images_path': 'images',
         'masks_path': 'masks',
         'images': images,
         'masks': masks,
         'mean_pixel': mean_pixel,
      }

      with open(train_json_path, 'w') as f:
         json.dump(train_json, f, indent = 3)

      return train_json

   def generate_style_patches(self, train_patches):
      target_path = self.get_config_path('style_patches_target_path')
      crop_dimension = train_patches['crop_dimension']
      count = len(train_patches['images'])
      style_json_path = os.path.join(target_path, 'style.json')

      if os.path.isfile(style_json_path):
         with open(style_json_path, 'r') as f:
            style_json = json.load(f)
         if len(style_json['images']) == count and style_json['crop_dimension'] == crop_dimension:
            return style_json
         else:
            shutil.rmtree(target_path)

      metadata = self.get_metadata()
      images_dir = self.get_config_path('images_dir')
      images = [Image(os.path.join(images_dir, filename), distance) for filename, distance in metadata.items()]
      target_path = os.path.join(target_path, 'images')
      ensure_dir(target_path)

      executor = ThreadPoolExecutor(max_workers=self.config['workers'])
      jobs = []

      for i, image in enumerate(random.choices(images, k=count)):
         jobs.append(executor.submit(image.generate_random_crop, target_path, crop_dimension, prefix=i))

      patches = []

      for job in as_completed(jobs):
         patches.append(job.result())

      style_json = {
         'images_path': 'images',
         'images': patches,
         'crop_dimension': crop_dimension,
      }

      with open(style_json_path, 'w') as f:
         json.dump(style_json, f, indent = 3)

      return style_json

