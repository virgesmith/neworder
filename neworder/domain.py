from os import stat
import numpy as np

class Domain:

  # Edge behaviour
  WRAP = 1
  CONSTRAIN = 2
  #BOUNCE = 3

  def __init__(self, dim, edge, continuous):
    self.__dim = dim
    assert edge in [Domain.WRAP, Domain.CONSTRAIN]
    self.__edge = edge
    self.__continuous = continuous

  @property
  def dim(self):
    return self.__dim

  @property
  def edge(self):
    return self.__edge

  @property
  def continuous(self):
    return self.__continuous

class Space(Domain):

  @classmethod
  def unbounded(cls, dim):
    assert dim 
    return cls(np.full(dim, -np.inf), np.full(dim, +np.inf))

  def __init__(self, min, max, edge = Domain.CONSTRAIN):
    assert len(min) and len(min) == len(max)
    super().__init__(len(min), edge, True)
    #self.dim = len(min)
    self.min = min
    self.max = max

  @property
  def extent(self):
    return self.min, self.max

  def move(self, points, delta, ungroup=False):
    # group tuples into a single array if necessary
    if type(points) == tuple:
      points = np.column_stack(points)
    if type(delta) == tuple:
      delta = np.column_stack(delta)

    assert points.dtype == float
    assert delta.dtype == float
    assert points.shape[-1] == self.dim and delta.shape[-1] == self.dim
    if self.edge == Domain.CONSTRAIN:
      result = np.clip(points + delta, self.min, self.max)
    else:
      result = self.min + np.mod(points + delta - self.min, self.max - self.min)

    if ungroup:
      result = np.split(result, self.dim, axis=1)
    return result

  def dists2(self, points):
    # group tuples into a single array if necessary
    if type(points) == tuple:
      points = np.column_stack(points)
    assert points.dtype == float
    n = points.shape[0]
    d = points.shape[1]
    d2 = np.zeros((n,n))
    for i in range(d):
      d2 += (np.tile(points[:,i],n).reshape((n,n)) - np.repeat(points[:,i],n).reshape(n,n))**2
    return d2

  def dists(self, points):
    return np.sqrt(self.dists2(points))

  def __repr__(self):
    return "%s dim=%d min=%s max=%s edge=%s" % (self.__class__.__name__, self.dim, self.min, self.max, self.edge)

class Grid(Domain):
  def __init__(self, ext, edge = Domain.CONSTRAIN):
    assert len(ext) and ext.dtype == np.int64
    super().__init__(len(ext), edge, False)
    self.__extent = ext

  @property
  def extent(self):
    return self.__extent

  def move(self, points, delta):
    # group tuples into a single array if necessary
    if type(points) == tuple:
      points = np.column_stack(points)
    # group tuples into a single array if necessary
    if type(delta) == tuple:
      points = np.column_stack(delta)
    assert points.dtype == np.int64
    assert delta.dtype == np.int64
    assert points.shape[-1] == self.dim and delta.shape[-1] == self.dim
    if self.edge == Domain.CONSTRAIN:
      return np.clip(points + delta, self.min, self.max)
    else:
      return np.mod(points + delta - self.min, self.extent)

  def dists2(self, points, to_points=None):
    # group tuples into a single array if necessary
    if type(points) == tuple:
      points = np.column_stack(points)
    if type(to_points) == tuple:
      to_points = np.column_stack(to_points)
    # distances w.r.t. self if to_points not explicitly specified
    if to_points is None:
      to_points = points
    n = points.shape[0]
    m = to_points.shape[0]
    d = points.shape[1]
    dmatrix = np.zeros((m,n), dtype=np.int64)
    for i in range(d):
      # for non-diagonal neighbours only, distance in hops
      #dmatrix += np.abs(np.tile(points[:,i],m).reshape((m,n)) - np.repeat(to_points[:,i],n).reshape(m,n))
      dmatrix += (np.tile(points[:,i],m).reshape((m,n)) - np.repeat(to_points[:,i],n).reshape(m,n))**2
    return dmatrix

  def dists(self, points, to_points=None):
    return np.sqrt(self.dists2(points, to_points))


