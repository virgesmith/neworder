import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Animation:

  def __init__(self, x, y):
    fig, self.ax = plt.subplots()
    self.xdata, self.ydata = [], []
    self.ln, = plt.plot([], [], 'ro', animated=True)
    self.y = y
    ani = FuncAnimation(fig, self.__update, frames=x, init_func=self.__init, blit=True)
    plt.show()

  def __init(self):
    self.ax.set_xlim(0, len(self.y))
    self.ax.set_ylim(0, 1000)
    return self.ln,

  def __update(self, frame):
    self.xdata.append(frame)
    self.ydata.append(self.y[frame])
    self.ln.set_data(self.xdata, self.ydata)
    return self.ln,

# x = np.array(range(101))
# y = (x + 1) * x / 10

# a = Animation(x, y)