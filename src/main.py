import sys
from Dataset import Dataset
from Unknot import Unknot

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

annotation_patches = source.generate_annotation_patches(scale_transfer_target=target, max_workers=10)
annotation_patches = target.apply_style_transfer(annotation_patches, device='cuda', max_workers=10, steps=10000)

# unknot = Unknot()
# unknot.prepare_datasets(device='cuda', max_workers=10)
