"""
Example of how to draw a map from a shapefile (or part thereof)
"""

import geopandas 
import pandas as pd
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
  # see also /mnt/16.04/home/az/.ukboundaries/cache/
  df = geopandas.read_file("../Mistral/data/england_msoa_2011.shp")
  df = df[df.label.str.contains(lad_match)].to_crs(epsg=3857)

  if not len(df):
    print("No polygons found")
    return

  ax = df.plot(figsize=(10, 10), alpha=0.3, edgecolor='k')
  # url="https://a.basemaps.cartocdn.com/light_all/tileZ/tileX/tileY.png"
  # url="https://a.tile.openstreetmap.org/tileZ/tileX/tileY.png"
  add_basemap(ax, df.crs) #, 
  plt.show()

if __name__ == "__main__":
  lad_match = "E08000032|E18000035"
  main(lad_match)