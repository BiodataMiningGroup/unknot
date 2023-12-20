import sys
import os

from src.Dataset import Dataset
from src.ObjectDetector import ObjectDetector

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])
results_dir = sys.argv[3]

config = os.path.join(os.path.dirname(__file__), 'src/faster_rcnn_r50_fpn_1x.py')

detector = ObjectDetector(config, source, target, results_dir, workers=4, batch_size=16)
checkpoint_path = detector.train()

detector.detect(checkpoint_path)
