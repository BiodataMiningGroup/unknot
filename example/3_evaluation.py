import sys
import os
import json

from src.Dataset import Dataset

dataset = Dataset(sys.argv[1])

evaluation = dataset.evaluate_test_images(sys.argv[2])
print(json.dumps(evaluation, indent = 3))
