from os import stat
import numpy as np

def _bounce(point, min, max):
  for d in range(len(point)):
    if point[d] < min[d]: point[d] = min[d] - point[d]
    if point[d] > max[d]: point[d] = max[d] - point[d]
  return point

class Domain:

  # Edge behaviour
  UNBOUNDED = 0
  WRAP = 1
  CONSTRAIN = 2
  BOUNCE = 3

  def __init__(self, dim, edge, continuous):
    self.__dim = dim
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
  """ Continuous rectangular n-dimensional domain """

  @staticmethod
  def unbounded(dim):
    assert dim
    return Space(np.full(dim, -np.inf), np.full(dim, +np.inf), edge=Domain.UNBOUNDED)

  def __init__(self, min, max, edge=Domain.CONSTRAIN):
    assert len(min) and len(min) == len(max)
    super().__init__(len(min), edge, edge)

    # Space supports all edge behaviours
    assert edge in [Domain.UNBOUNDED, Domain.WRAP, Domain.CONSTRAIN, Domain.BOUNCE]

    self.min = min
    self.max = max

  @property
  def extent(self):
    return self.min, self.max

  def move(self, positions, velocities, delta_t, ungroup=False):
    """ Returns translated positions AND velocities """
    # group tuples into a single array if necessary
    if type(positions) == tuple:
      positions = np.column_stack(positions)
    if type(velocities) == tuple:
      velocities = np.column_stack(velocities)

    assert positions.dtype == float
    assert velocities.dtype == float
    assert positions.shape[-1] == self.dim and velocities.shape[-1] == self.dim
    if self.edge == Domain.UNBOUNDED: 
      p = positions + velocities * delta_t
      v = velocities
    elif self.edge == Domain.CONSTRAIN: 
      p = positions + velocities * delta_t
      v = velocities
      hitmin = np.where(p < self.min, 1, 0)
      p = np.where(hitmin, self.min, p)
      v = np.where(hitmin, 0, v)
      hitmax = np.where(p > self.max, 1, 0)
      p = np.where(hitmax, self.max, p)
      v = np.where(hitmax, 0, v)   
    elif self.edge == Domain.BOUNCE:
      p = positions + velocities * delta_t
      v = velocities
      hitmin = np.where(p < self.min, 1, 0)
      p = np.where(hitmin, 2*self.min - p, p)
      v = np.where(hitmin, -v, v)
      hitmax = np.where(p > self.max, 1, 0)
      p = np.where(hitmax, 2*self.max - p, p)
      v = np.where(hitmax, -v, v)   
    else:
      p = self.min + np.mod(positions + velocities * delta_t - self.min, self.max - self.min)
      v = velocities

    if ungroup:
      p = np.split(p, self.dim, axis=1)
      v = np.split(v, self.dim, axis=1)
    return p, v

  def dists2(self, points, to_points):
    # group tuples into a single array if necessary
    if type(points) == tuple:
      points = np.column_stack(points)
    if type(to_points) == tuple:
      to_points = np.column_stack(to_points)
    # distances w.r.t. self if to_points not explicitly specified
    if to_points is None:
      to_points = points
    assert points.dtype == float
    assert to_points.dtype == float
    n = points.shape[0]
    m = to_points.shape[0]
    d = points.shape[1]
    d2 = np.zeros((m,n))
    if self.edge != Domain.WRAP:
      for i in range(d):
        d2 += (np.tile(points[:,i],m).reshape((m,n)) - np.repeat(to_points[:,i],n).reshape(m,n))**2
    else: # wrapped domains need special treatment - distance across an edge may be less than naive distance
      for i in range(d):
        d1d = np.abs(np.tile(points[:,i],m).reshape((m,n)) - np.repeat(to_points[:,i],n).reshape(m,n))
        r = self.max[i] - self.min[i]
        d1d = np.where(d1d > r/2, r - d1d, d1d)
        d2 += d1d*d1d

    return d2

  def dists(self, points, to_points=None):
    return np.sqrt(self.dists2(points, to_points))

  def __repr__(self):
    return "%s dim=%d min=%s max=%s edge=%s" % (self.__class__.__name__, self.dim, self.min, self.max, self.edge)

class Grid(Domain):
  """ Discrete rectangular n-dimensional domain """

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


