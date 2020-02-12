import sys
import os

from src.Dataset import Dataset
from src.ObjectDetection import ObjectDetector
import src.mrcnn.utils as utils

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])
results = sys.argv[3]

annotation_patches = source.get_annotation_patches()

detector = ObjectDetector(os.path.join(results, 'model'))

config = {
   'RPN_NMS_THRESHOLD': 0.85,
   'IMAGES_PER_GPU': 5,
}

train_scheme = [
   {
      'layers': 'heads',
      'epochs': 10,
      'learning_rate': 0.001
   },
   {
      'layers': 'heads',
      'epochs': 10,
      'learning_rate': 0.0005
   },
   {
      'layers': 'heads',
      'epochs': 10,
      'learning_rate': 0.0001
   },
   {
      'layers': 'all',
      'epochs': 10,
      'learning_rate': 0.0001
   },
   {
      'layers': 'all',
      'epochs': 10,
      'learning_rate': 0.00005
   },
   {
      'layers': 'all',
      'epochs': 10,
      'learning_rate': 0.00001
   },
]

coco_model_path = 'mask_rcnn_coco.h5'
if not os.path.exists(coco_model_path):
   utils.download_trained_weights(coco_model_path)

detector.perform_training(annotation_patches, scheme=train_scheme, config=config, initial_model=coco_model_path)

detector.perform_inference(annotation_patches, target, os.path.join(results, 'detections'))
