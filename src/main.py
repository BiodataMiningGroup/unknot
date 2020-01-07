import sys
from Dataset import Dataset

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

source.generate_scale_transfer(target)
