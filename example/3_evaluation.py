import sys
import os
import json

from src.Dataset import Dataset

dataset = Dataset(sys.argv[1])
results_path = os.path.join(sys.argv[2], 'detections')

evaluation = dataset.evaluate_test_images(results_path)
print(json.dumps(evaluation, indent = 3))
