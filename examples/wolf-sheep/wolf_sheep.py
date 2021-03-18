
import numpy as np
import pandas as pd
import neworder as no

import matplotlib.pyplot as plt

PAUSE=0.001

WOLF_COLOUR = "black"
SHEEP_COLOUR = "red"
GRASS_COLOUR = "green"

class WolfSheep(no.Model):

  def __init__(self, params):

    # hard-coded to unit timestep
    super().__init__(no.LinearTimeline(0.0, 1.0), no.MonteCarlo.deterministic_independent_stream)

    self.width = params["grid"]["width"]
    self.height = params["grid"]["height"]
    n_wolves = params["wolves"]["starting_population"]
    n_sheep =params["sheep"]["starting_population"]

    self.wolf_reproduce = params["wolves"]["reproduce"]
    self.sheep_reproduce = params["sheep"]["reproduce"]
    self.wolf_speed = params["wolves"]["speed"]
    self.sheep_speed = params["sheep"]["speed"]

    self.wolf_gain_from_food = params["wolves"]["gain_from_food"]
    self.sheep_gain_from_food = params["sheep"]["gain_from_food"]
    self.grass_regrowth_time = params["grass"]["regrowth_time"]

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
      index=pd.Index(name="id", data=no.df.unique_index(n_wolves)),
      data={
        "x": self.mc().ustream(n_wolves) * self.width,
        "y": self.mc().ustream(n_wolves) * self.height,
        "energy": (self.mc().ustream(n_wolves) + self.mc().ustream(n_wolves)) * self.wolf_gain_from_food
      }
    )
    self.__assign_cell(self.wolves)

    self.sheep = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(n_sheep)),
      data={
        "x": self.mc().ustream(n_sheep) * self.width,
        "y": self.mc().ustream(n_sheep) * self.height,
        "energy": (self.mc().ustream(n_sheep) + self.mc().ustream(n_sheep)) * self.sheep_gain_from_food
      }
    )
    self.__assign_cell(self.sheep)

    self.wolf_pop = [len(self.wolves)]
    self.sheep_pop = [len(self.sheep)]
    self.grass_prop = [100.0 * self.grass.fully_grown.mean()]
    self.t = [self.timeline().index()]

    (self.ax_g, self.ax_w, self.ax_s, self.ax_t0, self.ax_wt, self.ax_st, self.ax_t1, self.ax_gt) = self.__init_plot()

    # no.log(self.wolves)
    # no.log(self.sheep)
    # no.log(self.grass)

  def step(self):

    # step each population
    self.__step_grass()
    self.__step_wolves()
    self.__step_sheep()

  def check(self):
    # record data
    self.t.append(self.timeline().index())
    self.wolf_pop.append(len(self.wolves))
    self.sheep_pop.append(len(self.sheep))
    self.grass_prop.append(100.0 * self.grass.fully_grown.mean())

    self.__update_plot()

    # no.log("wolves=%d sheep=%d grass=%f" % (len(self.wolves),
    #                                         len(self.sheep),
    #                                         100.0 * len(self.grass[self.grass.fully_grown]) / (self.width *self.height)))
    if self.timeline().index() > 300: self.halt()
    if self.wolves.empty:
      no.log("Wolves have died out")
    if self.sheep.empty:
      no.log("Sheep have died out")
      self.halt()
    return True

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
    figs = plt.figure(constrained_layout=True, figsize=(10,5))
    gs = figs.add_gridspec(2, 2)
    ax0 = figs.add_subplot(gs[:, 0])
    ax1 = figs.add_subplot(gs[0, 1])
    ax2 = figs.add_subplot(gs[1, 1])

    # agent map
    ax_g = ax0.imshow(np.flip(self.grass.countdown.values.reshape(self.height, self.width), axis=0),
      extent=[0, self.width, 0, self.height], cmap="Greens_r", alpha=0.5)
    ax_w = ax0.scatter(self.wolves.x, self.wolves.y, s=6, color=WOLF_COLOUR)
    ax_s = ax0.scatter(self.sheep.x, self.sheep.y, s=6, color=SHEEP_COLOUR)
    ax0.set_axis_off()

    # wolf and sheep population
    ax_wt = ax1.plot(self.t, self.wolf_pop, color=WOLF_COLOUR)
    ax_st = ax1.plot(self.t, self.sheep_pop, color=SHEEP_COLOUR)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, max(self.wolf_pop[0], self.sheep_pop[0])])
    #ax1.set_xlabel("Step")
    ax1.legend(["Wolves", "Sheep"])

    # grass
    ax_gt = ax2.plot(0, self.grass_prop[0], color=GRASS_COLOUR)
    ax2.set_xlim([0, 100])
    ax2.set_ylim([0.0, 100.0])
    ax2.legend(["% fully grown grass"])
    ax2.set_xlabel("Step")

    plt.tight_layout()
    plt.pause(PAUSE)
    return ax_g, ax_w, ax_s, ax1, ax_wt, ax_st, ax2, ax_gt

  def __update_plot(self):
    self.ax_g.set_data(np.flip(self.grass.countdown.values.reshape(self.height, self.width), axis=0))
    self.ax_w.set_offsets(np.c_[self.wolves.x, self.wolves.y])
    self.ax_s.set_offsets(np.c_[self.sheep.x, self.sheep.y])

    self.ax_wt[0].set_data(self.t, self.wolf_pop)
    self.ax_st[0].set_data(self.t, self.sheep_pop)
    self.ax_t0.set_xlim([0,self.t[-1]])
    self.ax_t0.set_ylim([0,max(max(self.wolf_pop), max(self.sheep_pop))])

    self.ax_gt[0].set_data(self.t, self.grass_prop)
    self.ax_t1.set_xlim([0,self.t[-1]])

    plt.pause(PAUSE)
    #plt.savefig("/tmp/wolf-sheep%04d.png" % self.timeline().index(), dpi=80)

