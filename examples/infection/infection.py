from typing import Any
from enum import Enum
from time import sleep
import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
from shapely import line_interpolate_point
import geopandas as gpd

import neworder as no

class Status(Enum):
  SUSCEPTIBLE = 0
  INFECTED = 1
  IMMUNE = 2
  DEAD = 3

  @property
  def rgba(self) -> tuple[float, float, float, float]:
    match self:
      case Status.SUSCEPTIBLE:
        return (1.0, 1.0, 1.0, 1.0)
      case Status.INFECTED:
        return (1.0, 0.0, 0.0, 1.0)
      case Status.IMMUNE:
        return (0.0, 1.0, 0.0, 1.0)
      case Status.DEAD:
        return (0.0, 0.0, 0.0, 1.0)


class Infection(no.Model):
  def __init__(self,
               point: tuple[float, float],
               dist: float,
               n_agents: int,
               n_infected: int,
               speed: float,
               infection_radius: float,
               recovery_time: int,
               mortality: float) -> None:
    super().__init__(no.LinearTimeline(0.0, 1.0), no.MonteCarlo.deterministic_independent_stream)
    # expose the model's MC engine to numpy
    self.nprand = no.as_np(self.mc)
    # create the spatial domain
    self.domain = no.GeospatialGraph.from_point(point, dist, network_type="drive", crs='epsg:27700')

    # set the parameters
    self.infection_radius = infection_radius
    self.recovery_time = recovery_time
    self.marginal_mortality = 1.0 - (1.0 - mortality) ** (1.0 / recovery_time)

    # create the agent data, which is stored in a geopandas geodataframe
    start_positions = self.domain.all_nodes.sample(n=n_agents, random_state=self.nprand, replace=True).index.values
    speeds = self.nprand.lognormal(np.log(speed), 0.2, n_agents)
    agents = gpd.GeoDataFrame(data={"node": start_positions, "speed": speeds, "status": Status.SUSCEPTIBLE, "t_infect": no.time.never()})
    agents["dest"] = agents["node"].apply(self.__random_next_dest)
    agents["path"] = agents[["node", "dest"]].apply(lambda r: self.domain.shortest_path(r["node"], r["dest"], weight="length"), axis=1)
    agents["dist"] = agents.path.apply(lambda p: p.length)
    agents["offset"] = 0.0
    agents["geometry"] = agents["path"].apply(lambda linestr: line_interpolate_point(linestr, 0))
    infected = self.nprand.choice(agents.index, n_infected, replace=False)
    agents.loc[infected, "status"] = Status.INFECTED
    agents.loc[infected, "t_infect"] = self.timeline.index()

    self.agents = agents
    self.fig, self.g = self.__init_visualisation()

  def step(self) -> None:
    self.__update_position()
    self.__infect_nearby()
    self.__recover()
    self.__succumb()
    num_infected = (self.agents.status == Status.INFECTED).sum()
    num_immune = (self.agents.status == Status.IMMUNE).sum()
    num_dead = (self.agents.status == Status.DEAD).sum()
    self.__update_visualisation(num_infected, num_immune, num_dead)
    if num_infected == 0:
        sleep(5)
        self.halt()
        self.finalise()

  def finalise(self) -> None:
    no.log(f"total steps: {self.timeline.index()}")
    no.log(f"infections: {len(self.agents.t_infect.dropna())}")
    no.log(f"recoveries: {(self.agents.status == Status.IMMUNE).sum()}")
    no.log(f"deaths: {(self.agents.status == Status.DEAD).sum()}")
    no.log(f"unaffected: {(self.agents.status == Status.SUSCEPTIBLE).sum()}")

  def __random_next_dest(self, node: int) -> int:
    # ensure dest is different from origin
    dest = node
    while dest == node:
      dest = self.domain.all_nodes.sample(n=1, random_state=self.nprand).index.values[0]
    return dest

  def __update_position(self) -> None:
    self.agents.offset += self.agents.speed
    # move agent along its route
    self.agents["geometry"] = self.agents[["path", "offset"]].apply(lambda r: line_interpolate_point(r["path"], r["offset"]), axis=1)
    # check if arrived at destination and set a new destination if necessary
    overshoots = self.agents.offset >= self.agents.dist
    if not overshoots.empty:
      # offset <- offset - dist
      self.agents.loc[overshoots, "offset"] -= self.agents.loc[overshoots, "dist"]
      # node <- dest
      self.agents.loc[overshoots, "node"] = self.agents.loc[overshoots, "dest"]
      # dest <- random
      self.agents.loc[overshoots, "dest"] = self.agents.loc[overshoots, "node"].apply(self.__random_next_dest)
      # path <- (node, dest), dist <- new_dist
      self.agents.loc[overshoots, "path"] = self.agents.loc[overshoots, ["node", "dest"]] \
        .apply(lambda r: self.domain.shortest_path(r["node"], r["dest"], weight="length"), axis=1)
      self.agents.loc[overshoots, "dist"] = self.agents.loc[overshoots, "path"].apply(lambda p: p.length)
      # finally update position
      self.agents.loc[overshoots, "geometry"] = self.agents.loc[overshoots, "path"].apply(lambda linestr: line_interpolate_point(linestr, 0))

  def __infect_nearby(self) -> None:
    infected = self.agents[self.agents.status == Status.INFECTED].geometry
    susceptible = self.agents[self.agents.status == Status.SUSCEPTIBLE].geometry
    new_infections = []
    # loop over smallest group for efficiency
    if len(infected) < len(susceptible):
      for i in infected:
        new = susceptible.geometry.distance(i) < self.infection_radius
        # new[new].index gives us only the index values corresponding to True
        new_infections.extend(new[new].index)
    else:
      for i, p in susceptible.items():
        new = infected.geometry.distance(p) < self.infection_radius
        if new.any():
          new_infections.append(i)
    self.agents.loc[new_infections, "status"] = Status.INFECTED
    self.agents.loc[new_infections, "t_infect"] = self.timeline.index()

  def __recover(self) -> None:
    t = self.timeline.index()
    self.agents.loc[(t - self.agents.t_infect >= self.recovery_time) & (self.agents.status == Status.INFECTED), "status"] = Status.IMMUNE

  def __succumb(self) -> None:
    infected = self.agents[self.agents.status == Status.INFECTED]
    death = self.mc.hazard(self.marginal_mortality, len(infected)).astype(bool)
    self.agents.loc[infected[death].index.values, "status"] = Status.DEAD
    self.agents.loc[infected[death].index.values, "speed"] = 0.0

  def __init_visualisation(self) -> tuple[Any, Any]:
    plt.ion()
    fig, ax = ox.plot_graph(self.domain.graph, bgcolor="w", node_size=5, edge_linewidth=2, edge_color="#777777", figsize=(12,9))
    plt.tight_layout()
    # optionally add a basemap:
    # import contextily as ctx
    # ctx.add_basemap(ax, crs=self.domain.crs, url=ctx.providers.OpenTopoMap)
    g = ax.scatter(self.agents.geometry.x, self.agents.geometry.y, color=self.agents.status.apply(lambda c: c.rgba), edgecolor='k')
    fig.suptitle("[q to quit]")
    fig.canvas.mpl_connect('key_press_event', lambda event: self.halt() if event.key == "q" else None)
    fig.canvas.flush_events()
    return fig, g

  def __update_visualisation(self, num_infected, num_immune, num_dead) -> None:
    offsets = np.array(list(zip(self.agents.geometry.x, self.agents.geometry.y)))
    colours = self.agents.status.apply(lambda c: c.rgba)
    self.g.set_offsets(offsets)
    self.g.set_facecolors(colours)
    self.fig.suptitle(f"step {self.timeline.index()}: inf={num_infected} imm={num_immune} dead={num_dead} / {len(self.agents)} [q to quit]")

    self.fig.canvas.flush_events()

