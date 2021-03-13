
import numpy as np
import pandas as pd
import neworder as no

import matplotlib.pyplot as plt

PAUSE=0.001

WOLF_COLOUR = "black"
SHEEP_COLOUR = "red"
class WolfSheep(no.Model):

  def __init__(self, width, height, n_wolves, n_sheep):

    # hard-coded to unit timestep
    super().__init__(no.LinearTimeline(0.0, 1.0), no.MonteCarlo.nondeterministic_stream)

    self.width = width
    self.height = height
    self.n_wolves = n_wolves
    self.n_sheep = n_sheep

    self.wolf_reproduce = 0.05
    self.sheep_reproduce = 0.04
    self.wolf_speed = 1.4
    self.sheep_speed = 0.9

    self.wolf_gain_from_food = 20
    self.sheep_gain_from_food = 4
    self.grass_regrowth_time = 30

    ncells = self.width * self.height
    self.grass = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(ncells)),
      data={
        "x": np.tile(np.arange(self.width) + 0.5, self.height),
        "y": np.repeat(np.arange(self.height) + 0.5, self.width),
        "fully_grown": self.mc().sample(ncells, [0.5, 0.5]).astype(bool), # equal likelihood
        "countdown": (self.mc().ustream(ncells) * self.grass_regrowth_time).astype(int)
      }
    )
    # adjust grass countdown (to max) for fully grown
    self.grass.loc[self.grass.fully_grown, "countdown"] = self.grass_regrowth_time

    self.wolves = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(self.n_wolves)),
      data={
        "x": self.mc().ustream(self.n_wolves) * self.width,
        "y": self.mc().ustream(self.n_wolves) * self.height,
        "energy": (self.mc().ustream(self.n_wolves) + self.mc().ustream(self.n_wolves)) * self.wolf_gain_from_food
      }
    )
    self.__assign_cell(self.wolves)

    self.sheep = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(self.n_sheep)),
      data={
        "x": self.mc().ustream(self.n_sheep) * self.width,
        "y": self.mc().ustream(self.n_sheep) * self.height,
        "energy": (self.mc().ustream(self.n_sheep) + self.mc().ustream(self.n_sheep)) * self.sheep_gain_from_food
      }
    )
    self.__assign_cell(self.sheep)

    (self.ax_g, self.ax_w, self.ax_s, self.ax_t, self.ax_wt, self.ax_st) = self.__init_plot()

    self.wolf_pop = [len(self.wolves)]
    self.sheep_pop = [len(self.sheep)]
    self.grass_prop = [self.grass.fully_grown.mean()]
    self.t = [self.timeline().index()]

    # no.log(self.wolves)
    # no.log(self.sheep)
    # no.log(self.grass)

  def step(self):

    # step each population
    self.__step_grass()
    self.__step_wolves()
    self.__step_sheep()

    # record data
    self.t.append(self.timeline().index()+1)
    self.wolf_pop.append(len(self.wolves))
    self.sheep_pop.append(len(self.sheep))
    self.grass_prop.append(self.grass.fully_grown.mean())

    self.__update_plot()

    # no.log("wolves=%d sheep=%d grass=%f" % (len(self.wolves),
    #                                         len(self.sheep),
    #                                         100.0 * len(self.grass[self.grass.fully_grown]) / (self.width *self.height)))
    #if self.timeline().index() > 200: self.halt()
    if self.wolves.empty:
      no.log("Wolves have died out")
    if self.sheep.empty:
      no.log("Sheep have died out")
      self.halt()

  def __step_grass(self):
    # grow grass
    self.grass.countdown -= 1
    just_grown = (~self.grass.fully_grown) & (self.grass.countdown <= 0)
    self.grass.loc[just_grown, "fully_grown"] = True
    self.grass.loc[just_grown, "countdown"] = self.grass_regrowth_time

  def __step_wolves(self):
    # move wolves (wrapped) and update cell
    self.wolves.x += (2 * self.mc().ustream(len(self.wolves)) - 1.0) * self.wolf_speed
    self.wolves.x = self.wolves.x % self.width
    self.wolves.y += (2 * self.mc().ustream(len(self.wolves)) - 1.0) * self.wolf_speed
    self.wolves.y = self.wolves.y % self.height
    self.wolves.energy -= 1
    self.__assign_cell(self.wolves)

    # eat sheep if available
    # no.log(np.intersect1d(self.sheep.cell, self.wolves.cell))
    # no.log(self.sheep[self.sheep.cell.isin(self.wolves.cell)])
    diners = self.wolves.loc[self.wolves.cell.isin(self.sheep.cell)]
    self.wolves.loc[self.wolves.cell.isin(self.sheep.cell), "energy"] += self.wolf_gain_from_food
    # NB *all* the sheep in cells with wolves get eaten (or at least killed)
    self.sheep = self.sheep[~self.sheep.cell.isin(diners.cell)]

    # remove dead
    self.wolves = self.wolves[self.wolves.energy >= 0]

    # breed
    m = self.mc().hazard(self.wolf_reproduce, len(self.wolves))
    self.wolves.loc[m == 1, "energy"] /= 2
    cubs = self.wolves[m == 1].copy().set_index(no.df.unique_index(int(sum(m))))
    self.wolves = self.wolves.append(cubs)

  def __step_sheep(self):
    # move sheep (wrapped)
    self.sheep.x += (2 * self.mc().ustream(len(self.sheep)) - 1.0) * self.sheep_speed
    self.sheep.x = self.sheep.x % self.width
    self.sheep.y += (2 * self.mc().ustream(len(self.sheep)) - 1.0) * self.sheep_speed
    self.sheep.y = self.sheep.y % self.height
    self.sheep.energy -= 1
    self.__assign_cell(self.sheep)

    # eat grass if available
    self.sheep.energy += self.grass.loc[self.sheep.cell, "fully_grown"].values * self.sheep_gain_from_food
    self.grass.loc[self.sheep.cell, "fully_grown"] = False

    # remove dead
    self.sheep = self.sheep[self.sheep.energy >= 0]

    # breed
    m = self.mc().hazard(self.sheep_reproduce, len(self.sheep))
    self.sheep.loc[m == 1, "energy"] /= 2
    lambs = self.sheep[m == 1].copy().set_index(no.df.unique_index(int(sum(m))))
    self.sheep = self.sheep.append(lambs)

  def __assign_cell(self, agents):
    # not ints for some reason
    agents["cell"] = (agents.x.astype(int) + self.width * agents.y.astype(int)).astype(int)

  def __init_plot(self):
    fig, axs = plt.subplots(1,2, figsize=(10,5))

    ax_g = axs[0].imshow(np.flip(self.grass.countdown.values.reshape(self.height, self.width), axis=0),
      extent=[0, self.width, 0, self.height], cmap="Greens_r", alpha=0.5)
    #plt.scatter(self.grass.x, self.grass.y, marker="o", s=40, c=self.grass.countdown, cmap="Greens")
    ax_w = axs[0].scatter(self.wolves.x, self.wolves.y, color=WOLF_COLOUR)
    ax_s = axs[0].scatter(self.sheep.x, self.sheep.y, color=SHEEP_COLOUR)
    axs[0].set_axis_off()

    ax_wt = axs[1].plot(0,len(self.wolves), color=WOLF_COLOUR)
    ax_st = axs[1].plot(0,len(self.sheep), color=SHEEP_COLOUR)
    axs[1].set_xlim([0, 100])
    axs[1].set_ylim([0, max(self.n_wolves, self.n_sheep)])
    axs[1].set_xlabel("Step")
    axs[1].legend(["Wolves", "Sheep"])
    plt.tight_layout()
    plt.pause(PAUSE)
    return ax_g, ax_w, ax_s, axs[1], ax_wt, ax_st

  def __update_plot(self):
    self.ax_g.set_data(np.flip(self.grass.countdown.values.reshape(self.height, self.width), axis=0))
    self.ax_w.set_offsets(np.c_[self.wolves.x, self.wolves.y])
    self.ax_s.set_offsets(np.c_[self.sheep.x, self.sheep.y])

    #self.ax_t.append((self.timeline().index(), len(self.wolves)))
    #self.ax_t.lines[0].get_xydata().append([self.timeline().index(), len(self.wolves)])
    self.ax_wt[0].set_data(self.t, self.wolf_pop)
    self.ax_st[0].set_data(self.t, self.sheep_pop)
    self.ax_t.set_xlim([0,self.t[-1]])
    self.ax_t.set_ylim([0,max(max(self.wolf_pop), max(self.sheep_pop))])
    #print(self.ax_t[0].get_data())
    # self.ax_t.lines[0].set_xdata(list(range(self.timeline().index())))
    #self.ax_t[0].set_xdata(list(range(self.timeline().index())))
    #self.ax_t[0].set_ydata(self.wolf_pop)
    #stop
    plt.pause(PAUSE)

