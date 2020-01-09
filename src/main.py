import sys
import os

from Dataset import Dataset
from ObjectDetection import ObjectDetector

import mrcnn.utils

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

# Torch and TensorFlow can't be executed in the same script.
# annotation_patches = source.generate_annotation_patches(scale_transfer_target=target, max_workers=10)
# annotation_patches = target.apply_style_transfer(annotation_patches, device='cuda', max_workers=10, steps=10000)

annotation_patches = source.get_annotation_patches()

detector = ObjectDetector('model')

config = {
   'RPN_NMS_THRESHOLD': 0.85,
   'IMAGES_PER_GPU': 5,
}

train_scheme = [
   {
      'layers': 'heads',
      'epochs': 1,
      'learning_rate': 0.001
   },
   # {
   #    'layers': 'heads',
   #    'epochs': 20,
   #    'learning_rate': 0.001
   # },
   # {
   #    'layers': 'heads',
   #    'epochs': 20,
   #    'learning_rate': 0.0005
   # },
   # {
   #    'layers': 'heads',
   #    'epochs': 20,
   #    'learning_rate': 0.0001
   # },
   # {
   #    'layers': 'all',
   #    'epochs': 20,
   #    'learning_rate': 0.0001
   # },
   # {
   #    'layers': 'all',
   #    'epochs': 20,
   #    'learning_rate': 0.00005
   # },
   # {
   #    'layers': 'all',
   #    'epochs': 20,
   #    'learning_rate': 0.00001
   # },
]

coco_model_path = 'mask_rcnn_coco.h5'
if not os.path.exists(coco_model_path):
   mrcnn.utils.download_trained_weights(coco_model_path)

detector.perform_training(annotation_patches, scheme=train_scheme, config=config, initial_model=coco_model_path)
