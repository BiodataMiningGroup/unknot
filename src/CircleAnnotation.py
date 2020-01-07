import numpy as np

class CircleAnnotation(object):
   def __init__(self, row):
      self.label = row[2]
      points = np.array(row[13].strip('[]').split(','), dtype=float)
      self.center = points[:2]
      self.radius = points[2]
      self.scale = 1

   def set_scale(self, factor):
      self.scale = factor

   def get_center(self):
      return tuple(np.round(self.center * self.scale).astype(int))

   def get_radius(self):
      return int(np.round(self.radius * self.scale))
