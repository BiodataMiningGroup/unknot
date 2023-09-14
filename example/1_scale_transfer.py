import sys
import os

from src.Dataset import Dataset

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

source.generate(target)
