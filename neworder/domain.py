import numpy as np

class Domain:
  pass

class Space(Domain):
  def __init__(self, min, max, wrap = False):
    assert len(min) and len(min) == len(max)
    self.dim = len(min)
    self.min = min
    self.max = max
    self.wrap = wrap

  def dim(self):
    return self.dim

  def extent(self):
    return self.min, self.max

  def move(self, points, delta):
    assert points.shape[-1] == self.dim and delta.shape[-1] == self.dim
    if not self.wrap:
      return np.clip(points + delta, self.min, self.max)
    else:
      return self.min + np.mod(points + delta - self.min, self.max - self.min)

  def dists2(self, points):
    n = points.shape[0]
    d = points.shape[1]
    d2 = np.zeros((n,n))
    for i in range(d):
      d2 += (np.tile(points[:,i],n).reshape((n,n)) - np.repeat(points[:,i],n).reshape(n,n))**2
    return d2

  def dists(self, points):
    return np.sqrt(self.dists2(points))

  def __repr__(self):
    return "%s dim=%d min=%s max=%s wrap=%s" % (self.__class__.__name__, self.dim, self.min, self.max, self.wrap)
