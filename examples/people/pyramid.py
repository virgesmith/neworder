
""" pyramid plots """

import matplotlib.pyplot as plt

import matplotlib.animation as anim

# see https://stackoverflow.com/questions/27694221/using-python-libraries-to-plot-two-horizontal-bar-charts-sharing-same-y-axis
def plot(ages, males, females):

  xmax = 5000 #max(max(males), max(females))

  fig, axes = plt.subplots(ncols=2, sharey=True)
  plt.gca().set_ylim([min(ages),max(ages)+1])
  fig.suptitle("2011")
  axes[0].set(title='Males')
  axes[0].set(xlim=[0, xmax])
  axes[1].set(title='Females')
  axes[1].set(xlim=[0, xmax])
  axes[0].yaxis.tick_right()
  #axes[1].set(yticks=ages)
  axes[0].invert_xaxis()
  for ax in axes.flat:
    ax.margins(0.03)
  fig.tight_layout()
  fig.subplots_adjust(wspace=0.125)
  mbar = axes[0].barh(ages, males, align='center', color='blue')
  fbar = axes[1].barh(ages, females, align='center', color='red')
  plt.pause(0.1)
  plt.ion()
  return fig, axes, mbar, fbar

def update(title, fig, axes, mbar, fbar, ages, males, females):
  # mbar.remove()
  # mbar = axes[0].barh(ages, males, align='center', color='blue')
  # fbar.remove()
  # fbar = axes[1].barh(ages, females, align='center', color='red')
  for rect, h in zip(mbar, males):
    rect.set_width(h)
  for rect, h in zip(fbar, females):
    rect.set_width(h)

  fig.suptitle(title)
  plt.pause(0.1)
  return mbar, fbar

def hist(a):
  plt.hist(a, bins=range(120))
  plt.show()

# class Pyramid:
#   def __init__(self, ages, males, females):

#     fig, axes, mbar, fbar = plot(ages, males[0], females[0])

#     self.anim = animation.FuncAnimation(fig, self.__animate, interval=100, frames=numbins, repeat=False)

#   def save(self, filename):
#     # there seems to be no way of preventing passing the loop once setting to the saved gif and it loops forever, which is very annoying
#     self.anim.save(filename, dpi=80, writer=animation.ImageMagickWriter(extra_args=["-loop", "1"]))

#   # def show(self):
#   #   plt.show()

#   def __animate(self, frameno):
#     i = 0
#     for rect, h in zip(self.patches, self.n):
#       rect.set_height(h if i <= frameno else 0)
#       i = i + 1
#     return self.patches


def animated(ages, males, females):
  fig, axes, mbar, fbar = plot(ages, males[0], females[0])

  def animate(i):
    #print(type(axes[0]), dir(axes[0]))
    #axes[0].remove()
    # for bar in fig.subplots():
    #   bar.remove()
    # mbar = axes[0].barh(ages, males[i], align='center', color='blue')
    # fbar = axes[1].barh(ages, females[i], align='center', color='red')
    for rect, y in zip(mbar, males[i]):
      rect.set_width(y)
    for rect, y in zip(fbar, females[i]):
      rect.set_width(y)

    #axes[0].set_data(males[i])
    #print(type(mbar), dir(mbar))
    #axes[0].set_height(males[i])
    fig.suptitle(str(i+2011))
    plt.pause(0.1)

    #update(str(i+2011), fig, axes, mbar, fbar, ages, males[i], females[i])

  animator = anim.FuncAnimation(fig, animate, frames=41, interval=1, repeat=False)

  animator.save("./pyramid.gif", dpi=80, writer=anim.FFMpegFileWriter())
  #ImageMagickWriter(extra_args=["-loop", "1"]))

  plt.show()
  animator.save("./pyramid.gif", dpi=80, writer=anim.FFMpegFileWriter())
  #plt.pause(1)
