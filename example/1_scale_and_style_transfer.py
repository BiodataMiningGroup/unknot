import sys
import os

from src.Dataset import Dataset

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

# Get number of usable CPUs.
workers = len(os.sched_getaffinity(0))

# Torch and TensorFlow can't be executed in the same script so style transfer and object
# detection have to be split into two separate scripts.
annotation_patches = source.generate_annotation_patches(scale_transfer_target=target, max_workers=workers, reuse_patches=True)
annotation_patches = target.apply_style_transfer(annotation_patches, device='cuda', max_workers=workers, steps=10000, reuse_patches=True)
