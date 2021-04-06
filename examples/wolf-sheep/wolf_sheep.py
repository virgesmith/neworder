
import numpy as np
import pandas as pd
import neworder as no

import matplotlib.pyplot as plt

PAUSE=1e-6

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
    self.init_wolf_speed = params["wolves"]["speed"]
    self.init_sheep_speed = params["sheep"]["speed"]
    self.wolf_speed_stddev = np.sqrt(params["wolves"]["speed_variance"])
    self.sheep_speed_stddev = np.sqrt(params["sheep"]["speed_variance"])

    self.wolf_gain_from_food = params["wolves"]["gain_from_food"]
    self.sheep_gain_from_food = params["sheep"]["gain_from_food"]
    self.grass_regrowth_time = params["grass"]["regrowth_time"]

    ncells = self.width * self.height
    self.grass = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(ncells)),
      data={
        "x": np.tile(np.arange(self.width) + 0.5, self.height),
        "y": np.repeat(np.arange(self.height) + 0.5, self.width),
        # 50% initial probability of being fully grown, other states uniform
        "countdown": self.mc().sample(ncells, [0.5] + [0.5/(self.grass_regrowth_time-1)]*(self.grass_regrowth_time-1))
      }
    )

    self.wolves = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(n_wolves)),
      data={
        "x": self.mc().ustream(n_wolves) * self.width,
        "y": self.mc().ustream(n_wolves) * self.height,
        "speed": self.init_wolf_speed,
        "energy": (self.mc().ustream(n_wolves) + self.mc().ustream(n_wolves)) * self.wolf_gain_from_food
      }
    )
    self.__assign_cell(self.wolves)

    self.sheep = pd.DataFrame(
      index=pd.Index(name="id", data=no.df.unique_index(n_sheep)),
      data={
        "x": self.mc().ustream(n_sheep) * self.width,
        "y": self.mc().ustream(n_sheep) * self.height,
        "speed": self.init_sheep_speed,
        "energy": (self.mc().ustream(n_sheep) + self.mc().ustream(n_sheep)) * self.sheep_gain_from_food
      }
    )
    self.__assign_cell(self.sheep)

    self.wolf_pop = [len(self.wolves)]
    self.sheep_pop = [len(self.sheep)]
    self.grass_prop = [100.0 * len(self.grass[self.grass.countdown==0])/len(self.grass)]
    self.wolf_speed = [self.wolves.speed.mean()]
    self.sheep_speed = [self.sheep.speed.mean()]
    self.wolf_speed_var = [self.wolves.speed.var()]
    self.sheep_speed_var = [self.sheep.speed.var()]
    self.t = [self.timeline().index()]

    (self.ax_g, self.ax_w, self.ax_s,
     self.ax_t1, self.ax_wt, self.ax_st,
     self.ax_t2, self.ax_gt,
     self.ax_t3, self.ax_ws, self.ax_ss,
     self.ax_t4, self.ax_wv, self.ax_sv) = self.__init_plot()

    # no.log(self.wolves)
    # no.log(self.sheep)
    # no.log(self.grass)
    self.paused = False

    # seed numpy random generator using our generator (for reproducible normal samples)
    self.npgen = np.random.default_rng(self.mc().raw())

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
    self.grass_prop.append(100.0 * len(self.grass[self.grass.countdown==0])/len(self.grass))
    self.wolf_speed.append(self.wolves.speed.mean())
    self.sheep_speed.append(self.sheep.speed.mean())
    self.wolf_speed_var.append(self.wolves.speed.var())
    self.sheep_speed_var.append(self.sheep.speed.var())

    self.__update_plot()

    if self.wolves.empty:
      no.log("Wolves have died out")
    if self.sheep.empty:
      no.log("Sheep have died out")
      self.halt()
    return True

  def __step_grass(self):
    # grow grass
    self.grass.countdown = np.clip(self.grass.countdown-1, 0, None)

  def __step_wolves(self):
    # move wolves (wrapped) and update cell
    self.wolves.x += (2 * self.mc().ustream(len(self.wolves)) - 1.0) * self.wolves.speed
    self.wolves.x = self.wolves.x % self.width
    self.wolves.y += (2 * self.mc().ustream(len(self.wolves)) - 1.0) * self.wolves.speed
    self.wolves.y = self.wolves.y % self.height
    # half of energy (initially) is consumed by moving
    self.wolves.energy -= 0.5 + 0.5 * self.wolves.speed / self.init_wolf_speed
    self.__assign_cell(self.wolves)

    # eat sheep if available
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
    # evolve speed/burn rate from mother + random factor
    cubs.speed += self.npgen.normal(0.0, self.wolf_speed_stddev, len(cubs))
    self.wolves = self.wolves.append(cubs)

  def __step_sheep(self):
    # move sheep (wrapped)
    self.sheep.x += (2 * self.mc().ustream(len(self.sheep)) - 1.0) * self.sheep.speed
    self.sheep.x = self.sheep.x % self.width
    self.sheep.y += (2 * self.mc().ustream(len(self.sheep)) - 1.0) * self.sheep.speed
    self.sheep.y = self.sheep.y % self.height
    # half of energy (initially) is consumed by moving
    self.sheep.energy -= 0.5 + 0.5 * self.sheep.speed / self.init_sheep_speed
    self.__assign_cell(self.sheep)

    # eat grass if available
    grass_available = self.grass.loc[self.sheep.cell]
    self.sheep.energy += (grass_available.countdown.values == 0) * self.sheep_gain_from_food
    self.grass.loc[self.sheep.cell, "countdown"] = self.grass.loc[self.sheep.cell, "countdown"].apply(lambda c: self.grass_regrowth_time if c == 0 else c)

    # remove dead
    self.sheep = self.sheep[self.sheep.energy >= 0]

    # breed
    m = self.mc().hazard(self.sheep_reproduce, len(self.sheep))
    self.sheep.loc[m == 1, "energy"] /= 2
    lambs = self.sheep[m == 1].copy().set_index(no.df.unique_index(int(sum(m))))
    # evolve speed from mother + random factor
    lambs.speed += self.npgen.normal(0.0, self.sheep_speed_stddev, len(lambs))
    self.sheep = self.sheep.append(lambs)

  def __assign_cell(self, agents):
    # not ints for some reason
    agents["cell"] = (agents.x.astype(int) + self.width * agents.y.astype(int)).astype(int)

  def __init_plot(self):
    self.figs = plt.figure(figsize=(15,5))
    self.figs.suptitle("[q to quit]", y=0.05, x= 0.05)
    gs = self.figs.add_gridspec(2, 3)
    ax0 = self.figs.add_subplot(gs[:, 0])
    ax1 = self.figs.add_subplot(gs[0, 1])
    ax2 = self.figs.add_subplot(gs[1, 1])

    ax3 = self.figs.add_subplot(gs[0, 2])
    ax4 = self.figs.add_subplot(gs[1, 2])

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
    ax1.set_ylabel("Population")
    ax1.legend(["Wolves", "Sheep"])

    # grass
    ax_gt = ax2.plot(0, self.grass_prop[0], color=GRASS_COLOUR)
    ax2.set_xlim([0, 100])
    ax2.set_ylim([0.0, 100.0])
    ax2.set_ylabel("% fully grown grass")
    ax2.set_xlabel("Step")

    # wolf and sheep speed
    ax_ws = ax3.plot(self.t, self.wolf_speed, color=WOLF_COLOUR)
    ax_ss = ax3.plot(self.t, self.sheep_speed, color=SHEEP_COLOUR)
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0, max(self.wolf_speed[0], self.sheep_speed[0])])
    ax3.set_ylabel("Average speed")

    # wolf and sheep speed variance
    ax_wv = ax4.plot(self.t, self.wolf_speed_var[0], color=WOLF_COLOUR)
    ax_sv = ax4.plot(self.t, self.sheep_speed_var[0], color=SHEEP_COLOUR)
    ax4.set_xlim([0, 100])
    ax4.set_ylim([0, max(self.wolf_speed_var[0], self.sheep_speed_var[0])])
    ax4.set_ylabel("Speed variance")
    ax4.set_xlabel("Step")

    plt.tight_layout()
    #plt.pause(PAUSE)
    plt.ion()
    #plt.show()

    def on_keypress(event):
      if event.key == "q":
        self.halt()

    self.figs.canvas.mpl_connect('key_press_event', on_keypress)

    return ax_g, ax_w, ax_s, \
           ax1, ax_wt, ax_st, \
           ax2, ax_gt, \
           ax3, ax_ws, ax_ss, \
           ax4, ax_wv, ax_sv


  def __update_plot(self):
    self.ax_g.set_data(np.flip(self.grass.countdown.values.reshape(self.height, self.width), axis=0))
    self.ax_w.set_offsets(np.c_[self.wolves.x, self.wolves.y])
    self.ax_s.set_offsets(np.c_[self.sheep.x, self.sheep.y])

    def __update_axis(ax, s0, s1, t, d0, d1):
      s0[0].set_data(t, d0)
      s1[0].set_data(t, d1)
      ax.set_xlim([0,t[-1]])
      ax.set_ylim([0,max(max(d0), max(d1))])

    __update_axis(self.ax_t1, self.ax_wt, self.ax_st, self.t, self.wolf_pop, self.sheep_pop)

    self.ax_gt[0].set_data(self.t, self.grass_prop)
    self.ax_t2.set_xlim([0,self.t[-1]])

    __update_axis(self.ax_t3, self.ax_ws, self.ax_ss, self.t, self.wolf_speed, self.sheep_speed)
    __update_axis(self.ax_t4, self.ax_wv, self.ax_sv, self.t, self.wolf_speed_var, self.sheep_speed_var)

    plt.pause(PAUSE)

    #plt.savefig("/tmp/wolf-sheep%04d.png" % self.timeline().index(), dpi=80)

