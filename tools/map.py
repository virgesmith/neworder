#!/usr/bin/env python3
"""
Example of how to draw a map from a shapefile (or part thereof)
"""

import pandas as pd
import geopandas 
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

def add_basemap(ax, crs, url='http://a.tile.stamen.com/terrain/tileZ/tileX/tileY.png'):
  xmin, xmax, ymin, ymax = ax.axis()
  bb = geopandas.GeoSeries([Point(xmin, ymin), Point(xmax, ymax)])
  bb.crs = crs
  bb = bb.to_crs(epsg=4326)
  zoom = ctx.calculate_zoom(bb[0].x, bb[0].y, bb[1].x, bb[1].y)
  basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
  ax.imshow(basemap, extent=extent, interpolation='bilinear')
  # restore original x/y limits
  ax.axis((xmin, xmax, ymin, ymax))

def main(lad_match):
  # 4326 is lat/lon
  # 3857 is mercator
  df = geopandas.read_file("../Mistral/data/Local_Authority_Districts_December_2016_Ultra_Generalised_Clipped_Boundaries_in_Great_Britain.shp")
  print(df.columns.values)
  df = df[df.lad16cd.str.contains(lad_match)].to_crs(epsg=3857)

  df["unoblongity"] = df.st_areasha / df.st_lengths ** 2

  if not len(df):
    print("No polygons found")
    return


  ax = df.plot(figsize=(10, 10), alpha=0.5, column='unoblongity', cmap='Greens', edgecolor='k', ax=None)
  # url="https://a.basemaps.cartocdn.com/light_all/tileZ/tileX/tileY.png" requires attribution: "Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
  # url="https://a.tile.openstreetmap.org/tileZ/tileX/tileY.png"
  add_basemap(ax, df.crs, url="https://tiles.wmflabs.org/bw-mapnik/tileZ/tileX/tileY.png")
  plt.title("GB LADs")
  plt.axis("off")
  plt.show()

if __name__ == "__main__":
  lad_match = ".*" #"E0|W0|S1"
  main(lad_match)