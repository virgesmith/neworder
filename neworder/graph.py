from __future__ import annotations
from typing import Any, Generator
import networkx as nx  # type: ignore[import]
import osmnx as ox  # type: ignore[import]
from shapely.ops import linemerge  # type: ignore[import]
from shapely.geometry import LineString, MultiLineString, Polygon  # type: ignore[import]
import geopandas as gpd  # type: ignore[import]


"""Spatial domains that are defined by graphs/networks"""
class GeospatialGraph:

  def __init__(self, G: nx.Graph, crs: str | None = None) -> None:
    if crs:
      self.__graph = ox.project_graph(G, to_crs=crs)
    else:
      self.__graph = G
    self.__nodes, self.__edges = ox.graph_to_gdfs(self.__graph)

  @classmethod
  def from_point(cls, point: tuple[float, float], *args: Any, crs: str | None = None,  **kwargs: Any) -> GeospatialGraph:
    G = ox.graph_from_point(point, *args, **kwargs)
    return cls(G, crs)

  @property
  def crs(self) -> str:
    return self.__graph.graph["crs"]

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

  def subgraph(self, origin: int, **kwargs: Any) -> nx.Graph:
    return nx.ego_graph(self.__graph, origin, **kwargs)

  def isochrone(self, origin: int, **kwargs: Any) -> Polygon:
    subgraph = nx.ego_graph(self.__graph, origin, **kwargs)
    nodes, _ = ox.graph_to_gdfs(subgraph)
    return nodes.geometry.unary_union.convex_hull

  # # intersection
  # def __or__(self, other: GeospatialGraph | nx.Graph) -> GeospatialGraph:
  #   if isinstance(other, GeospatialGraph):
  #     other = other.__graph
  #   print(self.__graph.graph["crs"])
  #   self.__graph = nx.intersection(self.__graph, other)
  #   print(self.__graph.graph["crs"])
  #   self.__nodes, self.__edges = ox.graph_to_gdfs(other) # self.__graph)
  #   return self

  # # union
  # def __and__(self, other: GeospatialGraph | nx.Graph) -> GeospatialGraph:
  #   if isinstance(other, GeospatialGraph):
  #     other = other.__graph
  #   self.__graph = nx.union(self.__graph, other)
  #   self.__nodes, self.__edges = ox.graph_to_gdfs(self.__graph)
  #   return self


