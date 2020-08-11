
""" pyramid plots
"""

import matplotlib.pyplot as plt

# see https://stackoverflow.com/questions/27694221/using-python-libraries-to-plot-two-horizontal-bar-charts-sharing-same-y-axis
def plot(ages, males, females):

  xmax = max(max(males), max(females))

  fig, axes = plt.subplots(ncols=2, sharey=True)
  plt.gca().set_ylim([min(ages),max(ages)+1])
  axes[0].barh(ages, males, align='center', color='blue')
  axes[0].set(title='Males')
  axes[0].set(xlim=[0, xmax])
  axes[1].barh(ages, females, align='center', color='red')
  axes[1].set(title='Females')
  axes[1].set(xlim=[0, xmax])
  axes[0].yaxis.tick_right()
  #axes[1].set(yticks=ages)
  axes[0].invert_xaxis()
  for ax in axes.flat:
      ax.margins(0.03)
      ax.grid(True)
  fig.tight_layout()
  fig.subplots_adjust(wspace=0.125)
  plt.show()


# fig, axes = plt.subplots(ncols=2, sharey=True)
# axes[0].barh(y, staff, align='center', color='gray', zorder=10)
# axes[0].set(title='Number of sales staff')
# axes[1].barh(y, sales, align='center', color='gray', zorder=10)
# axes[1].set(title='Sales (x $1000)')

# axes[0].invert_xaxis()
# axes[0].set(yticks=y, yticklabels=states)
# axes[0].yaxis.tick_right()

# for ax in axes.flat:
#     ax.margins(0.03)
#     ax.grid(True)

# fig.tight_layout()
# fig.subplots_adjust(wspace=0.09)