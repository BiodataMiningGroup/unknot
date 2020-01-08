import sys
from Dataset import Dataset

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

train_patches = source.generate_scale_transfer(target)
style_patches = target.generate_style_patches(train_patches)
