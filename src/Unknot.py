import numpy as np
import random
import os
import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import HRNet.HRNet as HRNet
import HRNet.utils as utils

class Unknot(object):
   def __init__(self, source_dataset, target_dataset):
      self.source_dataset = source_dataset
      self.target_dataset = target_dataset
      self.scale_patches = None
      self.train_patches = None

   def perform_scale_transfer(self):
      self.scale_patches = self.source_dataset.generate_scale_transfer(self.target_dataset)

   def perform_style_transfer(self, device='cuda', steps=10000, show_every=100):
      if self.scale_patches is None:
         raise RuntimeError('You must perform scale transfer first.')

      style_patches = self.target_dataset.generate_style_patches(self.scale_patches)

      device = torch.device(device)

      VGG = models.vgg19(pretrained=True).features
      VGG.to(device)
      # only use VGG19 to extract features, we don't need to change it's parameters
      for parameter in VGG.parameters():
         parameter.requires_grad_(False)

      style_net = HRNet.HRNet()
      style_net.to(device)

      content_path = os.path.join(self.source_dataset.get_scale_transfer_path(), self.scale_patches['images_path'])
      content_paths = [os.path.join(content_path, file) for file in self.scale_patches['images']]
      # content_images = {path: utils.load_image(path) for path in content_paths}

      style_path = os.path.join(self.target_dataset.get_style_patches_path(), style_patches['images_path'])
      style_paths = [os.path.join(style_path, file) for file in style_patches['images']]
      # style_images = {path: utils.load_image(path) for path in style_paths}

      style_weights = {
         'conv1_1': 0.1,
         'conv2_1': 0.2,
         'conv3_1': 0.4,
         'conv4_1': 0.8,
         'conv5_1': 1.6,
      }

      content_weight = 150
      style_weight = 1

      optimizer = optim.Adam(style_net.parameters(), lr=5e-3)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

      print('Train HRNet')

      for epoch in range(0, steps + 1):
         content_path = random.choice(content_paths)
         content_image = utils.load_image(content_path).to(device)
         content_features = utils.get_features(content_image, VGG)

         style_path = random.choice(style_paths)
         style_image = utils.load_image(style_path).to(device)
         style_features = utils.get_features(style_image, VGG)
         style_gram_matrices = {layer: utils.get_grim_matrix(style_features[layer]) for layer in style_features}

         scheduler.step()

         target = style_net(content_image).to(device)
         target.requires_grad_(True)

         target_features = utils.get_features(target, VGG)
         content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

         style_loss = 0

         for layer in style_weights:
            target_feature = target_features[layer]
            target_gram_matrix = utils.get_grim_matrix(target_feature)
            style_gram_matrix = style_gram_matrices[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c * h * w)

         total_loss = content_weight * content_loss + style_weight * style_loss
         optimizer.zero_grad()
         total_loss.backward()
         optimizer.step()

         if epoch % show_every == 0:
            print("Epoch {} of {}:".format(epoch, steps))
            print('Total loss: ', total_loss.item())
            print('Content loss: ', content_loss.item())
            print('Style loss: ', style_loss.item())

      print('Apply HRNet')
      for path in content_paths:
         name = os.path.basename(path)
         print('Styling {}'.format(name))
         target = style_net(utils.load_image(path).to(device)).to(device)
         result = np.array(utils.im_convert(target) * 255).astype(np.uint8)
         self.source_dataset.store_style_transfer_patch(result, name)

   # TODO enable object detection on same dataset as source and target, without scale transfer and without style transfer
   def perform_object_detection(self):
      if self.train_patches is None:
         raise RuntimeError('You must perform style transfer first.')
