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

   def __init__(self, config_path):
      with open(config_path, 'r') as f:
         self.config = json.load(f)
      self.config_dir = os.path.dirname(config_path)
      self.name = self.config['name']
      self.crop_dimension = self.config.get('crop_dimension', 512)

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

   def get_style_patches_path(self):
      return self.get_config_path('style_patches_dir')

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

   def generate_style_patches(self, count, crop_dimension, max_workers=4, reuse_patches=True):
      patches = PatchesCollection.PatchesCollection(self.get_style_patches_path())

      if patches.exists:
         if reuse_patches and len(patches.images) == count and patches.crop_dimension == crop_dimension:
            return patches
         else:
            patches.destroy()

      patches.create(self.crop_dimension, has_masks=False)

      metadata = self.get_metadata()
      images_dir = self.get_config_path('images_dir')
      images = [Image.Image(os.path.join(images_dir, filename), distance) for filename, distance in metadata.items()]
      target_path = patches.get_images_path()

      executor = ThreadPoolExecutor(max_workers=max_workers)
      jobs = []

      for i, image in enumerate(random.choices(images, k=count)):
         jobs.append(executor.submit(image.generate_random_crop, target_path, crop_dimension, prefix=i))

      images = []

      for job in as_completed(jobs):
         images.append(job.result())

      patches.fill({
         'images': images,
         'crop_dimension': crop_dimension,
      })

      return patches

   def apply_style_transfer(self, source_patches, device='cuda', steps=10000, show_every=100, max_workers=4, reuse_patches=True):
      # Import torch locally becuase it is incompatible with TensorFlow which is imported
      # elsewhere.
      import torch
      from torchvision import models
      from .HRNet import HRNet
      from .HRNet import utils as HRNetUtils

      if not isinstance(source_patches, PatchesCollection.PatchesCollection):
         raise TypeError('The source patches must be a PatchesCollection.')

      if source_patches.for_dataset is not None and source_patches.for_dataset != self.name:
         raise RuntimeError('The source patches were generated for a different dataset.')

      if reuse_patches and hasattr(source_patches, 'style_transfer') and source_patches.style_transfer == True:
         return source_patches

      patch_count = len(source_patches.images)
      style_patches = self.generate_style_patches(patch_count, source_patches.crop_dimension, max_workers=max_workers, reuse_patches=reuse_patches)

      # The following code is adapted from:
      # https://github.com/limingcv/Photorealistic-Style-Transfer/blob/cd8919a529d406d27f157a065a7b84f0f3c1535b/Photorealistic%20Style%20Transfer/transfer.ipynb

      device = torch.device(device)

      VGG = models.vgg19(pretrained=True).features
      VGG.to(device)
      for parameter in VGG.parameters():
         parameter.requires_grad_(False)

      style_net = HRNet.HRNet()
      style_net.to(device)

      content_paths = source_patches.get_images_paths()
      style_paths = style_patches.get_images_paths()

      style_weights = {
         'conv1_1': 0.1,
         'conv2_1': 0.2,
         'conv3_1': 0.4,
         'conv4_1': 0.8,
         'conv5_1': 1.6,
      }

      content_weight = 150
      style_weight = 1

      optimizer = torch.optim.Adam(style_net.parameters(), lr=5e-3)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

      for epoch in range(0, steps + 1):
         content_path = random.choice(content_paths)
         content_image = HRNetUtils.load_image(content_path).to(device)
         content_features = HRNetUtils.get_features(content_image, VGG)

         style_path = random.choice(style_paths)
         style_image = HRNetUtils.load_image(style_path).to(device)
         style_features = HRNetUtils.get_features(style_image, VGG)
         style_gram_matrices = {layer: HRNetUtils.get_grim_matrix(style_features[layer]) for layer in style_features}

         scheduler.step()

         target = style_net(content_image).to(device)
         target.requires_grad_(True)

         target_features = HRNetUtils.get_features(target, VGG)
         content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

         style_loss = 0

         for layer in style_weights:
            target_feature = target_features[layer]
            target_gram_matrix = HRNetUtils.get_grim_matrix(target_feature)
            style_gram_matrix = style_gram_matrices[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c * h * w)

         total_loss = content_weight * content_loss + style_weight * style_loss
         optimizer.zero_grad()
         total_loss.backward()
         optimizer.step()

         if epoch % show_every == 0:
            print("Training HRNet epoch {} of {}".format(epoch, steps))

      mean_pixels = []

      for path in content_paths:
         print('Applying HRNet to {}'.format(os.path.basename(path)))
         target = style_net(HRNetUtils.load_image(path).to(device)).to(device)
         result = np.array(HRNetUtils.im_convert(target) * 255).astype(np.uint8)
         result = np.ascontiguousarray(result)
         mean_pixels.append(result.reshape((-1, 3)).mean(axis = 0))
         height, width, bands = result.shape
         image = VipsImage.new_from_memory(result.data, width, height, bands, 'uchar')
         image.write_to_file(path)

      mean_pixel = np.array(mean_pixels).mean(axis = 0).tolist()

      source_patches.fill({
         'style_transfer': True,
         'mean_pixel': mean_pixel,
      })

      return source_patches

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

