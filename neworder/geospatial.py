from __future__ import annotations

from typing import Any, Generator

try:
    import geopandas as gpd  # type: ignore[import]
    import networkx as nx  # type: ignore[import]
    import osmnx as ox  # type: ignore[import]
    from shapely.geometry import (  # type: ignore[import]
        LineString,
        MultiLineString,
        Polygon,
    )
    from shapely.ops import linemerge  # type: ignore[import]
except ImportError:
    raise ImportError(
        """optional dependencies are not installed.
Reinstalling neworder with the geospatial option should fix this:
pip install neworder[geospatial]"""
    )


class GeospatialGraph:
    """
    Spatial domains on Earth's surface that are defined by graphs/networks.
    Use of this class requires "geospatial" extras: pip install neworder[geospatial]
    """

    def __init__(self, G: nx.Graph, crs: str | None = None) -> None:
        if crs:
            self.__graph = ox.project_graph(G, to_crs=crs)
        else:
            self.__graph = G
        self.__nodes, self.__edges = ox.graph_to_gdfs(self.__graph)

    @classmethod
    def from_point(
        cls,
        point: tuple[float, float],
        *args: Any,
        crs: str | None = None,
        **kwargs: Any,
    ) -> GeospatialGraph:
        G = ox.graph_from_point(point, *args, **kwargs)
        return cls(G, crs)

    @property
    def crs(self) -> str:
        return self.__graph.graph["crs"]

    @property
    def graph(self) -> nx.MultiDiGraph | nx.Graph:
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
        route_segments = [self.__edges.loc[(nodes[i], nodes[i + 1], 0), "geometry"] for i in range(len(nodes) - 1)]
        return linemerge(MultiLineString(route_segments))

    def subgraph(self, origin: int, **kwargs: Any) -> nx.Graph:
        return nx.ego_graph(self.__graph, origin, **kwargs)

    def isochrone(self, origin: int, **kwargs: Any) -> Polygon:
        subgraph = nx.ego_graph(self.__graph, origin, **kwargs)
        nodes, _ = ox.graph_to_gdfs(subgraph)
        return nodes.geometry.unary_union.convex_hull
