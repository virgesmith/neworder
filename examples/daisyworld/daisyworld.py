
import neworder as no
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import signal

solar_luminosity = 0.8


class DaisyWorld(no.Model):

  EMPTY = 0
  WHITE_DAISY = 1
  BLACK_DAISY = 2

  MAX_AGE = 25

  DIFF_KERNEL = [
    [1.0 / 16, 1.0 / 16, 1.0 / 16],
    [1.0 / 16, 1.0 / 2,  1.0 / 16],
    [1.0 / 16, 1.0 / 16, 1.0 / 16]
  ]

  def __init__(self, gridsize, pct_white, pct_black):
    super().__init__(no.LinearTimeline(0, 1), no.MonteCarlo.deterministic_independent_stream)

    p = [pct_white, pct_black, 1 - pct_white - pct_black]
    init_pop = self.mc.sample(np.prod(gridsize), p).reshape(gridsize)

    self.domain = no.StateGrid(init_pop, edge=no.Domain.WRAP)
    self.age = (self.mc.ustream(self.domain.state.size) * DaisyWorld.MAX_AGE).astype(int).reshape(self.domain.state.shape)
    self.temperature = np.zeros(self.domain.state.shape)

    self.albedo = np.array([0.4, 0.75, 0.25])

    self.temperature = self.__calc_local_heating()
    self.__diffuse()

    print(self.domain.state)
    # print(self.age)
    # print(self.temperature)

    self.fig, self.img = self.__init_visualisation()

  def step(self):
    self.age += 1

    # update temperature
    self.temperature = 0.5 * (self.temperature + self.__calc_local_heating())
    self.__diffuse()
    no.log(f"mean temp = {np.mean(self.temperature)}")

    # update daisies
    self.age = np.where(
      np.logical_or(
        self.age >= DaisyWorld.MAX_AGE,
        self.domain.state == DaisyWorld.EMPTY),
      0, self.age)
    # kill old
    self.domain.state = np.where(self.age == 0, DaisyWorld.EMPTY, self.domain.state)

    # spawn new
    p_seed = np.clip(0.1457 * self.temperature - 0.0032 * self.temperature ** 2 - 0.6443, 0, 1)
    p_seed_white = np.where(self.domain.state == DaisyWorld.WHITE_DAISY, p_seed, 0)
    p_seed_black = np.where(self.domain.state == DaisyWorld.BLACK_DAISY, p_seed, 0)

    d = [self.timeline.index() % 3 - 1, self.timeline.index() // 3 % 3 - 1]

    new_white = np.logical_and(np.roll(self.mc.hazard(p_seed_white), d, axis=[0, 1]), self.domain.state == DaisyWorld.EMPTY)
    self.domain.state = np.where(new_white, DaisyWorld.WHITE_DAISY, self.domain.state)
    self.age = np.where(new_white, 0, self.age)

    new_black = np.logical_and(np.roll(self.mc.hazard(p_seed_black), d, axis=[0, 1]), self.domain.state == DaisyWorld.EMPTY)
    self.domain.state = np.where(new_black, DaisyWorld.BLACK_DAISY, self.domain.state)
    self.age = np.where(new_black, 0, self.age)

    # self.halt()
    # spawners = self.mc.hazard(p_seed_white)

    self.__update_visualisation()

    # sleep(0.1)

    if self.timeline.index() > 3000:
      self.halt()

  def __calc_local_heating(self):
    # local_heating = 0

    # get absorbed luminosity from state
    def fs(state):
      return (1.0 - self.albedo[state]) * solar_luminosity
    abs_lum = fs(self.domain.state)

    # get local heating from absorbed luminosity
    def fl(lum):
      return 72.0 * np.log(lum) + 80.0
    return fl(abs_lum)

  def __diffuse(self):
    padded = np.pad(self.temperature, pad_width=1, mode="wrap")
    self.temperature = signal.convolve(padded, DaisyWorld.DIFF_KERNEL, mode="same", method="direct")[1:-1, 1:-1]

  def __init_visualisation(self):

    # TODO copy wolf-sheep
    plt.ion()

    cmap = colors.ListedColormap(['blue', 'white', 'black'])

    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    img = plt.imshow(self.domain.state.T, cmap=cmap)
    plt.axis('off')
    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)
    fig.canvas.flush_events()

    return fig, img

  def __update_visualisation(self):
    self.img.set_array(self.domain.state.T)
    # plt.savefig("/tmp/daisyworld%04d.png" % self.timeline.index(), dpi=80)
    self.fig.canvas.flush_events()


if __name__ == "__main__":
  m = DaisyWorld((100, 100), 0.25, 0.2)
  no.run(m)
  # print(m.temperature)

