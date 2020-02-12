import sys
import os

from src.Dataset import Dataset

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

# Get number of usable CPUs.
workers = len(os.sched_getaffinity(0))

annotation_patches = source.generate_annotation_patches(scale_transfer_target=target, max_workers=workers, reuse_patches=True)
