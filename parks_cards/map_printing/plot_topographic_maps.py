"""
Playing around with the display of the N50 dataset.
"""

# %% import of packages and N50
import geopandas as gpd
import fiona
import json
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import shape, MultiLineString
from pathlib import Path
from scipy.interpolate import griddata

root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
N50_path = root_dir / "data/N50/Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"

layers = fiona.listlayers(N50_path)
for layer in layers:
    print(layer)
    pass


# %%
# National parks geometries
def print_national_park_names():
    national_parks_geometries_path = (
        root_dir / "data/card_details/national_parks_geo.json"
    )
    with open(national_parks_geometries_path) as f:
        national_parks_geometries = json.load(f)
    for park in national_parks_geometries.keys():
        print(park)
    return


def get_np_geo(np_name, new_crs=25833):
    national_parks_geometries_path = (
        root_dir / "data/card_details/national_parks_geo.json"
    )
    with open(national_parks_geometries_path) as f:
        national_parks_geometries = json.load(f)
    np_geo = shape(national_parks_geometries[np_name])
    np_gdf = gpd.GeoDataFrame(geometry=[np_geo])
    np_gdf.set_crs(epsg=4326, inplace=True)
    np_gdf.to_crs(epsg=new_crs, inplace=True)
    return np_gdf


def plot_N50_cover(
    cover_data: gpd.GeoDataFrame, figsize: tuple[float] = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the N50 cover data.

    Args:
        cover_data: gpd.GeoDataFrame - the N50 cover data
        figsize: tuple - the size of the figure (default: (10, 10))

    Returns:
        fig: plt.Figure - the figure object
        ax: plt.Axes - the axes object
    """
    if not figsize:
        figsize = (10, 10)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    legend = {
        "Alpinbakke": "khaki",
        "ÅpentOmråde": "lightgrey",
        "DyrketMark": "yellow",
        "Elv": "blue",
        "FerskvannTørrfall": "lightblue",
        "Gravplass": "grey",
        "Innsjø": "blue",
        "InnsjøRegulert": "blue",
        "Myr": "green",
        "Skog": "darkgreen",
        "SnøIsbre": "white",
        "SportIdrettPlass": "red",
        "Steinbrudd": "grey",
        "Steintipp": "grey",
        "Tettbebyggelse": "red",
        "Havflate": "red",
        "Industriområde": "red",
    }
    cover_data["color"] = cover_data["objtype"].map(legend)
    cover_data.plot(ax=ax, color=cover_data["color"])
    # Create a custom legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for label, color in legend.items()
    ]
    ax.legend(handles=handles, loc="upper left")
    plt.title("N50 Cover Data")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    return fig, ax


def interpolate_topographic_data(
    points: np.array, values: np.array, grid_size: int
) -> tuple[np.array]:
    """
    Interpolate the topographic data.

    Args:
        points: np.array (shape: (n, 2)) - the points of the topographic data
        values: np.array (shape: (n,)) - the values of the topographic data
        grid_size: int - the size of the grid

    Returns:
        grid_x: np.array (shape: (grid_size, grid_size)) - the x coordinates of the grid
        grid_y: np.array (shape: (grid_size, grid_size)) - the y coordinates of the grid
        grid_z: np.array (shape: (grid_size, grid_size)) - the z coordinates of the grid
    """
    x = points[:, 0]
    y = points[:, 1]
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size),
        np.linspace(y.min(), y.max(), grid_size),
    )
    grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")
    return grid_x, grid_y, grid_z


def convert_multilinestring_to_mesh(
    multilinestring_gdf: gpd.GeoDataFrame, grid_size: tuple[int] = (1000, 1000)
):
    """
    Convert a GeoDataFrame containing MultiLineString objects into a mesh grid.

    Args:
        multilinestring_gdf: gpd.GeoDataFrame - the GeoDataFrame containing MultiLineString objects
        grid_size: tuple of ints - the size of the grid

    Returns:
        grid_x: np.array (shape: (grid_size, grid_size)) - the x coordinates of the grid
        grid_y: np.array (shape: (grid_size, grid_size)) - the y coordinates of the grid
        grid_z: np.array (shape: (grid_size, grid_size)) - the z coordinates of the grid
    """
    points = []
    values = []

    for geom, value in zip(multilinestring_gdf.geometry, multilinestring_gdf["hoyde"]):
        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                points.extend(line.coords)
                values.extend([value] * len(line.coords))
        else:
            raise ValueError("Not a MultiLineString")

    points = np.array(points)
    values = np.array(values)

    grid_x, grid_y = np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size[0]),
        np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size[1]),
    )
    grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")

    return grid_x, grid_y, grid_z


def topographic_shaded_map(
    grid_x: np.array,
    grid_y: np.array,
    grid_z: np.array,
    figsize: tuple = None,
    topographic_cmap: str = "terrain",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a topographic map with shaded relief.
    cmap is is bound between -100 and 2700 (fits Norway's topography)

    Args:
        grid_x: np.array (shape: (n, m)) - the x coordinates of the grid
        grid_y: np.array (shape: (n, m)) - the y coordinates of the grid
        grid_z: np.array (shape: (n, m)) - the z coordinates of the grid
        figsize: tuple - the size of the figure (default: (10, 10))
        topographic_cmap: str - the colormap for the topographic map (default: "terrain")

    Returns:
        fig: plt.Figure - the figure object
        ax: plt.Axes - the axes object
    """
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    if not figsize:
        figsize = (10, 10)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        grid_z,
        cmap=topographic_cmap,
        vmin=-100,
        vmax=2700,
        extent=extent,
        origin="lower",
        alpha=1,
    )
    fig.colorbar(im, ax=ax, label="Altitude (m)")
    shaded = get_shading(grid_z, azimuth=45, altitude=315)
    plt.imshow(shaded, cmap="Greys", extent=extent, origin="lower", alpha=0.7)
    plt.title("Topographic Map with Shaded Relief")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    return fig, ax


def get_shading(elevation: np.array, azimuth: float, altitude: float) -> np.array:
    """
    Get the shading of the elevation data.

    Args:
        elevation: np.array (shape: (n, m)) - the elevation data
        azimuth: float - the azimuth angle in degrees
        altitude: float - the altitude angle in degrees

    Returns:
        shaded: np.array (shape: (n, m)) - the shaded elevation data
    """
    azimuth = np.deg2rad(azimuth)
    altitude = np.deg2rad(altitude)
    x, y = np.gradient(elevation)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))

    # -x here because of pixel orders in the SRTM tile
    aspect = np.arctan2(-x, y)

    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(
        slope
    ) * np.cos((azimuth - np.pi / 2.0) - aspect)

    return shaded


# %%

if __name__ == "__main__":
    # print_national_park_names()
    np_gdf = get_np_geo("Jotunheimen")
    np_bounds = tuple(np_geo.total_bounds)
    # extend bounds by 10km in each direction
    np_bounds = (
        np_bounds[0] - 10000,
        np_bounds[1] - 10000,
        np_bounds[2] + 10000,
        np_bounds[3] + 10000,
    )
    # %% N50 cover data
    cover_layer = "N50_Arealdekke_omrade"
    cover_gdf = gpd.read_file(N50_path, layer=cover_layer, bbox=np_bounds)
    print(cover_gdf["objtype"].unique())
    fig, ax = plot_N50_cover(cover_gdf, figsize=(20, 10))
    np_gdf.plot(ax=ax, color="none", edgecolor="red", linewidth=2)
    plt.show()
    # %% Topographic data
    # topographic_layer = "N50_Høyde_posisjon"
    topographic_layer = "N50_Høyde_senterlinje"
    topographic_gdf = gpd.read_file(N50_path, layer=topographic_layer, bbox=np_bounds)
    # points = np.array([(geom.x, geom.y) for geom in topographic_gdf.geometry])
    # values = topographic_gdf["hoyde"].values
    # grid_size = 1000
    # grid_x, grid_y, grid_z = interpolate_topographic_data(points, values, grid_size)
    grid_x, grid_y, grid_z = convert_multilinestring_to_mesh(
        topographic_gdf, grid_size=(1000, 1000)
    )
    fig, ax = topographic_shaded_map(grid_x, grid_y, grid_z, figsize=(20, 10))
    np_gdf.plot(ax=ax, color="none", edgecolor="red", linewidth=2)
    plt.show()

# %%
