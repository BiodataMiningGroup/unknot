import json
import csv
import numpy as np
import os
import random
import shutil
import math
from pyvips import Image as VipsImage
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import utils
from . import CircleAnnotation
from . import Image
from . import PatchesCollection

class Dataset(object):

   def __init__(self, config_path, only_classes=[]):
      with open(config_path, 'r') as f:
         self.config = json.load(f)
      self.config_dir = os.path.dirname(config_path)
      self.name = self.config['name']
      self.crop_dimension = self.config.get('crop_dimension', 512)
      self.only_classes = only_classes

   def get_config_path(self, key):
      return utils.normalize_path(self.config_dir, self.config[key])

   def get_metadata(self):
      metadata_path = self.get_config_path('metadata_file')
      with open(metadata_path, 'r') as file:
         reader = csv.reader(file)
         next(reader)
         metadata = {row[0]: float(row[1]) for row in reader}

      return metadata

   def get_mean_distance(self):
      metadata = self.get_metadata()

      return np.fromiter(metadata.values(), dtype=float).mean()

   def get_annotation_patches_path(self):
      return self.get_config_path('annotation_patches_dir')

   def should_ignore_annotation(self, annotation):
      return len(self.only_classes) > 0 and annotation.label not in self.only_classes

   def read_report(self, report_path):
      metadata = self.get_metadata()
      images_dir = self.get_config_path('images_dir')
      images = {}
      with open(report_path, 'r') as file:
         reader = csv.reader(file)
         next(reader)
         for row in reader:
            image_filename = row[8]
            annotation = CircleAnnotation.CircleAnnotation(row)

            if self.should_ignore_annotation(annotation):
               continue

            if image_filename not in images:
               image_path = os.path.join(images_dir, image_filename)
               image_distance = metadata[image_filename]
               images[image_filename] = Image.Image(image_path, image_distance)
            images[image_filename].add_annotation(annotation)

      return images.values()

   def get_train_images(self):
      train_report = self.get_config_path('train_annotations_file')

      return self.read_report(train_report)

   def get_test_images(self):
      test_report = self.get_config_path('test_annotations_file')

      return self.read_report(test_report)

   def get_annotation_patches(self):
      return PatchesCollection.PatchesCollection(self.get_annotation_patches_path())

   def generate_annotation_patches(self, scale_transfer_target=None, max_workers=4, reuse_patches=True):
      scale_transfer = scale_transfer_target is not None
      for_dataset = scale_transfer_target.name if scale_transfer else None

      if scale_transfer and not isinstance(scale_transfer_target, Dataset):
         raise TypeError('The scale transfer target dataset must be a Dataset.')

      patches = self.get_annotation_patches()

      if patches.exists:
         if reuse_patches and patches.scale_transfer == scale_transfer and patches.for_dataset == for_dataset and patches.crop_dimension == self.crop_dimension:
            return patches
         else:
            patches.destroy()

      patches.create(self.crop_dimension)
      images = self.get_train_images()
      images_target_path = patches.get_images_path()
      masks_target_path = patches.get_masks_path()

      if scale_transfer:
         target_distance = scale_transfer_target.get_mean_distance()
         for image in images:
            image.set_target_distance(target_distance)

      executor = ThreadPoolExecutor(max_workers=max_workers)
      jobs = []
      for image in images:
         jobs.append(executor.submit(image.generate_train_patches, images_target_path, masks_target_path, patches.crop_dimension))

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

      patches.fill({
         'scale_transfer': scale_transfer,
         'for_dataset': for_dataset,
         'images': images,
         'masks': masks,
         'mean_pixel': mean_pixel,
      })

      return patches

   def evaluate_test_images(self, results_path):
      images = self.get_test_images()

      evaluation = {
         'total_annotations': 0,
         'total_regions': 0,
         'total_detected_annotations': 0,
         'total_correct_regions': 0,
         'recall': 0,
         'precision': 0,
         'f-score': 0,
         'l-score': 0,
      }

      for image in images:
         ta, tr, cd, cr = image.evaluate(results_path)
         evaluation['total_annotations'] += ta
         evaluation['total_regions'] += tr
         evaluation['total_detected_annotations'] += cd
         evaluation['total_correct_regions'] += cr

      recall = evaluation['total_detected_annotations'] / evaluation['total_annotations']
      evaluation['recall'] = recall
      precision = evaluation['total_correct_regions'] / evaluation['total_regions']
      evaluation['precision'] = precision

      if recall > 0 and precision > 0:
         evaluation['f-score'] = 5 * recall * precision / (recall + 4 * precision)

      evaluation['l-score'] = 0.5 * (1 / (1 + math.exp(-0.25 * (recall * 100 - 80))) + 1 / (1 + math.exp(-0.5 * (precision * 100 - 10))))

      return evaluation

