import sys
from Dataset import Dataset
from Unknot import Unknot

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

unknot = Unknot(source, target)

unknot.perform_scale_transfer()
unknot.perform_style_transfer(device='cuda')
