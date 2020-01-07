import json
import csv
import numpy as np
from utils import normalize_path, ensure_dir
import os
from CircleAnnotation import CircleAnnotation
from Image import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

class Dataset(object):

   def __init__(self, config_path):
      with open(config_path, 'r') as f:
         self.config = json.load(f)
      self.config_dir = os.path.dirname(config_path)

   def get_metadata(self):
      metadata_path = normalize_path(self.config_dir, self.config['metadata'])
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
      images_dir = normalize_path(self.config_dir, self.config['images_dir'])
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
      train_report = normalize_path(self.config_dir, self.config['train_annotations_report'])

      return self.read_report(train_report)

   def read_test_report(self):
      test_report = normalize_path(self.config_dir, self.config['test_annotations_report'])

      return self.read_report(test_report)

   def generate_scale_transfer(self, target_dataset):
      if not isinstance(target_dataset, Dataset):
         raise TypeError('The target dataset must be a Dataset.')

      crop_dimension = self.config['crop_dim']
      target_path = normalize_path(self.config_dir, self.config['scale_transfer_target_path'])
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
         jobs.append(executor.submit(image.generate_train_patches, images_target_path, masks_target_path, crop_dimension))

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

      train_json_content = {
         'training_images_path': 'images',
         'training_masks_path': 'masks',
         'training_images': images,
         'training_masks': masks,
         'mean_pixel': mean_pixel,
         'crop_dimension': crop_dimension,
      }

      with open(os.path.join(target_path, 'train.json'), 'w') as f:
         json.dump(train_json_content, f, indent = 3)
