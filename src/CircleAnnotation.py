import numpy as np

class CircleAnnotation(object):
   def __init__(self, row):
      self.label = row[2]
      center, radius = self.get_center_and_radius_(row)
      self.center = center
      self.radius = radius
      self.scale = 1

   def set_scale(self, factor):
      self.scale = factor

   def get_center(self):
      return tuple(np.round(self.center * self.scale).astype(int).tolist())

   def get_radius(self):
      return int(np.round(self.radius * self.scale))

   def get_center_and_radius_(self, row):
      points = np.array(row[13].strip('[]').split(','), dtype=float)
      shape_id = int(row[11])
      if shape_id == 1: # Point
         # Points get the default radius of 50
         return points[:2], 50
      elif shape_id == 4: # Circle
         return points[:2], points[2]
      elif points.size > 0:
         points = points.reshape(-1, 2)
         x_max = points[:, 0].max()
         y_max = points[:, 1].max()
         x_min = points[:, 0].min()
         y_min = points[:, 1].min()
         center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])
         radius = np.NINF
         for c in points:
            radius = max(np.linalg.norm(center - c), radius)

         return center, radius
      else:
         raise Exception('Unsupported shape')

