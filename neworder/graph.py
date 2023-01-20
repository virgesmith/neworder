from typing import Any, Generator
import networkx as nx  # type: ignore[import]
import osmnx as ox  # type: ignore[import]
from shapely.ops import linemerge  # type: ignore[import]
from shapely.geometry import LineString, MultiLineString  # type: ignore[import]
import geopandas as gpd  # type: ignore[import]


"""Spatial domains that are defined by graphs/networks"""
class GeospatialGraph:
  def __init__(self, *args: Any, crs: str | None,  **kwargs: Any) -> None:
    G = ox.graph_from_point(*args, **kwargs)
    if crs:
      self.__graph = ox.project_graph(G, to_crs='epsg:27700')
    else:
      self.__graph = G
    self.__nodes, self.__edges = ox.graph_to_gdfs(self.__graph)

  @property
  def graph(self) -> gpd.GeoDataFrame:
    return self.__graph

  @property
  def all_nodes(self) -> gpd.GeoDataFrame:
    return self.__nodes

  @property
  def all_edges(self) -> gpd.GeoDataFrame:
    return self.__edges

  def edges_to(self, node: int) -> Generator[list[tuple[int, int]], None, None]:
    return self.__graph.in_edges(node)

  def edges_from(self, node: int) -> Generator[list[tuple[int, int]], None, None]:
    return self.__graph.out_edges(node)

  def shortest_path(self, origin: int, dest: int, **kwargs: Any) -> LineString:
    nodes = nx.shortest_path(self.__graph, origin, dest, **kwargs)
    route_segments = [self.__edges.loc[(nodes[i], nodes[i+1], 0), "geometry"] for i in range(len(nodes) - 1)]
    return linemerge(MultiLineString(route_segments))

  # TODO
  # def isochrone(self, origin, distance: float)


