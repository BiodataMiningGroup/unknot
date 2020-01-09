import sys
import os

from Dataset import Dataset

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

# Torch and TensorFlow can't be executed in the same script.
annotation_patches = source.generate_annotation_patches(scale_transfer_target=target, max_workers=10)
annotation_patches = target.apply_style_transfer(annotation_patches, device='cuda', max_workers=10, steps=10000)
