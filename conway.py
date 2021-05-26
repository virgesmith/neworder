import numpy as np
import pandas as pd
import neworder as no
import matplotlib.pyplot as plt
class Conway(no.Model):

  def __init__(self, nx, ny, n):
    super().__init__(no.LinearTimeline(0, 1, 100), no.MonteCarlo.deterministic_identical_stream)

    self.domain = no.domain.Grid(np.array([nx, ny]), edge=no.domain.Domain.WRAP)

    # pick n random positions
    rng = np.random.default_rng(self.mc().raw())
    index = no.df.unique_index(nx * ny)

    self.pop = pd.DataFrame(index=index, data={"x": index // nx, "y": index % nx, "s": 0})
    # randomly make n alive
    s = rng.choice(np.arange(nx*ny), n, replace=False)
    self.pop.loc[s, "s"] = 1

    self.fig, self.g = self.__init_visualisation()


  def step(self):
    occ = self.pop[self.pop.s==1]
    free = self.pop[self.pop.s==0]
    #no.log(occ)
    # get count of neigbours for occupied cells
    occ_neighbours = self.__neighbours((occ.x, occ.y), (self.pop.x, self.pop.y))
    free_neighbours = self.__neighbours((free.x, free.y), (self.pop.x, self.pop.y))

    # occ dies if <2 or >3 neighbours
    self.pop.loc[(self.pop.s==1) & (occ_neighbours < 2), "s"] = 0
    self.pop.loc[(self.pop.s==1) & (occ_neighbours > 3), "s"] = 0

    # free spawns if 3 neighbours
    self.pop.loc[(self.pop.s==0) & (free_neighbours == 3), "s"] = 1

    #no.log("occ=%d" % len(self.pop[self.pop.s==1]))

    self.__update_visualisation()

  def check(self):
    return True

  def __neighbours(self, points, ref_points=None):
    """ Gets a count of those in ref_points (or points) that are neighbours for each point in points, including diagonals """
    d2 = self.domain.dists2(points, ref_points)
    #print(d2 > 0)
    return np.sum(np.logical_and(d2 > 0.0, d2 <= 2.0), axis=1)

  def __init_visualisation(self):

    plt.ion()

    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    g = plt.scatter(self.pop[self.pop.s==1].x, self.pop[self.pop.s==1].y, s=20)
    # plt.xlim(0.0, self.range)
    # plt.ylim(0.0, self.range)
    plt.axis("off")

    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)

    fig.canvas.flush_events()

    return fig, g

  def __update_visualisation(self):

    self.g.set_offsets(np.c_[self.pop[self.pop.s==1].x, self.pop[self.pop.s==1].y])
    self.fig.canvas.flush_events()


m = Conway(50,50,1000)

no.run(m)
