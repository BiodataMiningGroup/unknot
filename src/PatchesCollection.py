import json
import shutil
import os
from . import utils

class PatchesCollection(object):
   def __init__(self, path):
      self.path = path
      self.json_path = os.path.join(self.path, 'patches.json')
      self.exists = os.path.isfile(self.json_path)
      self.patches_json = {}
      self.images = []
      self.masks = []

      if self.exists:
         with open(self.json_path, 'r') as f:
            self.patches_json = json.load(f)
            self.sync_patches_json()

   def create(self, crop_dimension, has_masks=True):
      if self.exists:
         raise RuntimeError('The patches collection already exists. Destroy it first.')

      self.patches_json = {
         'crop_dimension': crop_dimension,
      }

      utils.ensure_dir(self.get_images_path())
      if has_masks:
         utils.ensure_dir(self.get_masks_path())
      self.store_patches_json()
      self.sync_patches_json()
      self.exists = True

   def get_images_path(self):
      return os.path.join(self.path, 'images')

   def get_images_paths(self):
      base = self.get_images_path()

      return [os.path.join(base, file) for file in self.images]

   def get_masks_path(self):
      return os.path.join(self.path, 'masks')

   def get_masks_paths(self):
      base = self.get_masks_path()

      return [os.path.join(base, file) for file in self.masks]

   def store_patches_json(self):
      utils.ensure_dir(self.path)
      with open(self.json_path, 'w') as f:
         json.dump(self.patches_json, f, indent = 3)

   def sync_patches_json(self):
      for key, value in self.patches_json.items():
         setattr(self, key, value)

   def fill(self, values):
      self.patches_json.update(values)
      self.store_patches_json()
      self.sync_patches_json()

   def destroy(self):
      shutil.rmtree(self.path)
      for key in self.patches_json:
         if hasattr(self, key):
            delattr(self, key)
      self.patches_json = {}
      self.image = []
      self.masks = []
      self.exists = False
