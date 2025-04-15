"""
Playing around with the display of the N50 dataset.
"""

# %% import of packages and N50
import geopandas as gpd
import fiona
import json
import matplotlib.pyplot as plt
import numpy as np
import copy

from shapely.geometry import shape, MultiLineString, box, Polygon
from shapely import difference
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from shapely.ops import unary_union, polygonize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
N50_path = root_dir / "data/N50/Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"

layers = fiona.listlayers(N50_path)
for layer in layers:
    print(layer)
    pass


# %%
# National parks geometries
def get_national_parks_names():
    national_parks_geometries_path = (
        root_dir / "data/card_details/national_parks_geo.json"
    )
    with open(national_parks_geometries_path) as f:
        national_parks_geometries = json.load(f)
    return national_parks_geometries.keys()


def print_national_park_names():
    national_park_names = get_national_parks_names()
    for park in national_park_names:
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


def plot_protected_areas(
    exception: str, bounding_box: tuple, ax: plt.Axes, new_crs=25833
) -> plt.Axes:
    protected_areas_path = root_dir / "data/card_details/protected_areas_geo.json"
    with open(protected_areas_path) as f:
        protected_areas_geo = json.load(f)
    # make a copy of the bounding box and subtract all the protected areas from it
    white_out_box = copy.deepcopy(box(*bounding_box))
    for name, geo in protected_areas_geo.items():
        protected_geo = shape(geo)
        protected_gdf = gpd.GeoDataFrame(geometry=[protected_geo])
        protected_gdf.set_crs(epsg=4326, inplace=True)
        protected_gdf.to_crs(epsg=new_crs, inplace=True)
        protected_gdf = protected_gdf.cx[
            bounding_box[0] : bounding_box[2], bounding_box[1] : bounding_box[3]
        ]
        if len(protected_gdf) > 0:
            # Subtract the protected area from the white_out_box
            protected_gdf = protected_gdf.buffer(0)
            white_out_box = difference(white_out_box, protected_gdf.union_all())
            if name != exception:
                # protected_gdf.plot(
                #     ax=ax,
                #     color="none",
                #     edgecolor="indianred",
                #     hatch="///",
                #     # facecolor="indianred",  # Set the hatch color here
                #     linewidth=0.1,  # Width of the hatch line
                #     alpha=0.8,
                # )
                # Plot the border with the desired color
                protected_gdf.boundary.plot(
                    ax=ax,
                    edgecolor="darkgreen",
                    linewidth=2,  # Width of the outside line
                    linestyle=":",
                )

    # Plot the resulting whiteout box
    white_out_gdf = gpd.GeoDataFrame(geometry=[white_out_box])
    white_out_gdf.plot(ax=ax, color="white", edgecolor="none", alpha=0.3)
    return ax


def plot_roads(bbox: tuple, ax: plt.Axes, new_crs=25833):
    roads_layer = "N50_Samferdsel_senterlinje"
    roads_gdf = gpd.read_file(N50_path, layer=roads_layer, bbox=bbox)
    roads_gdf = roads_gdf.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    roads_gdf.plot(ax=ax, color="firebrick", linewidth=0.2, alpha=0.9)
    return ax


def plot_buildings(bbox: tuple, ax: plt.Axes, new_crs=25833):
    buildings_layer = "N50_BygningerOgAnlegg_omrade"
    buildings_gdf = gpd.read_file(N50_path, layer=buildings_layer, bbox=bbox)
    buildings_gdf = buildings_gdf.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    buildings_gdf.plot(ax=ax, color="firebrick", linewidth=0.2, alpha=0.9)
    return ax


def get_Norway_boundaries(goal_crs: int = 25833):
    layer = "N50_AdministrativeOmråder_grense"
    admin_gdf = gpd.read_file(N50_path, layer=layer)
    relevant_gdf = admin_gdf[
        admin_gdf["objtype"].isin(["Riksgrense", "Territorialgrense"])
    ]
    combined_lines = unary_union(relevant_gdf.geometry)
    polygons = list(polygonize(combined_lines))
    norway_boundary = unary_union(polygons)
    norway_boundary_gdf = gpd.GeoDataFrame(
        geometry=[norway_boundary], crs=admin_gdf.crs
    )
    norway_boundary_gdf.to_crs(epsg=goal_crs, inplace=True)
    return norway_boundary_gdf


def white_out_non_Norway(
    ax: plt.Axes, norway_boundary_gdf: gpd.GeoDataFrame, bounding_box: tuple
):
    box_coords = [
        (bounding_box[0], bounding_box[1]),
        (bounding_box[0], bounding_box[3]),
        (bounding_box[2], bounding_box[3]),
        (bounding_box[2], bounding_box[1]),
        (bounding_box[0], bounding_box[1]),
    ]
    white_out_box = Polygon(box_coords)
    for geom in norway_boundary_gdf.geometry:
        white_out_box = difference(white_out_box, geom)
    # white_out_box = difference(white_out_box, norway_boundary_gdf.geometry[0])
    white_out_gdf = gpd.GeoDataFrame(geometry=[white_out_box])
    white_out_gdf.plot(ax=ax, color="white", edgecolor="none", alpha=1)


def plot_location_in_Norway(
    np_geo: gpd.GeoDataFrame, norway_boundary_gdf: gpd.GeoDataFrame, ax: plt.Axes
):
    # Create an inset axis in the top left corner
    inset_ax = inset_axes(ax, width="20%", height="20%", loc="upper left")
    # Plot the Norway boundary and the national park location on the inset axis
    norway_boundary_gdf.plot(
        ax=inset_ax, edgecolor="crimson", facecolor=(1, 1, 1, 0.6), linewidth=3
    )
    np_geo.plot(ax=inset_ax, color="darkgreen", edgecolor="darkgreen", linewidth=8)
    # Remove axis labels and ticks from the inset plot
    inset_ax.axis("off")
    return ax


def plot_N50_cover(
    cover_data: gpd.GeoDataFrame,
    figsize: tuple[float] = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    plot_legend: bool = True,
    plot_title: str = "N50 Cover Data",
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
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cover_colors = {
        "freshwater": ["steelblue", 1],
        "trees": ["forestgreen", 0.8],
        "humans": ["firebrick", 0.9],
    }
    legend = {
        "Alpinbakke": ["tan", 0.2],
        "ÅpentOmråde": ["lightgrey", 0.1],
        "DyrketMark": ["gold", 0.3],
        "Myr": ["olive", 0.3],
        "Skog": ["forestgreen", 0.2],
        "SnøIsbre": ["white", 0.7],
        "FerskvannTørrfall": ["cyan", 0.5],
        "Havflate": ["lightsteelblue", 1],
        "Elv": cover_colors["freshwater"],
        "Innsjø": cover_colors["freshwater"],
        "InnsjøRegulert": cover_colors["freshwater"],
        "Myr": ["olive", 0.5],
        "Skog": ["forestgreen", 0.6],
        "SnøIsbre": ["white", 0.7],
        "Steinbrudd": ["grey", 0.2],
        "Steintipp": ["grey", 0.2],
        "Industriområde": cover_colors["humans"],
        "Lufthavn": cover_colors["humans"],
        "SportIdrettPlass": cover_colors["humans"],
        "Tettbebyggelse": cover_colors["humans"],
        "Golfbane": cover_colors["humans"],
        "Gravplass": cover_colors["humans"],
        "BymessigBebyggelse": cover_colors["humans"],
        "Park": cover_colors["humans"],
        "Rullebane": cover_colors["humans"],
    }
    object_types = cover_data["objtype"].unique()
    for objtype in object_types:
        if objtype not in legend.keys():
            print(f"Object type {objtype} not in legend")
    color_legend = {t: mcolors.to_rgb(c[0]) for t, c in legend.items()}
    # print(color_legend)
    alpha_legend = {t: c[1] for t, c in legend.items()}
    cover_data["color"] = cover_data["objtype"].map(color_legend)
    cover_data["alpha"] = cover_data["objtype"].map(alpha_legend)
    for objtype, group in cover_data.groupby("objtype"):
        group.plot(
            ax=ax,
            facecolor=group["color"].iloc[0],
            edgecolor="none",
            # linewidth=1.5,
            alpha=group["alpha"].iloc[0],
        )
    # Create a custom legend
    if plot_legend:
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
            for label, color in color_legend.items()
        ]
        ax.legend(handles=handles, loc="upper left")
    if plot_title:
        plt.title(plot_title)
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
    # print(len(multilinestring_gdf))
    for geom, value in zip(multilinestring_gdf.geometry, multilinestring_gdf["hoyde"]):
        if isinstance(geom, MultiLineString):
            for line in geom.geoms:
                points.extend(line.coords)
                values.extend([value] * len(line.coords))
        else:
            raise ValueError("Not a MultiLineString")

    points = np.array(points)
    # print(f"points shape: {points.shape}")
    values = np.array(values)

    grid_x, grid_y = np.meshgrid(
        np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size[0]),
        np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size[1]),
    )
    grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")
    min_value = values.min()
    max_value = values.max()
    return grid_x, grid_y, grid_z, min_value, max_value


def topographic_shaded_map(
    grid_x: np.array,
    grid_y: np.array,
    grid_z: np.array,
    figsize: tuple = None,
    topographic_cmap: str = "terrain",
    min_elevation: float = 0,
    max_elevation: float = 0,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    print_colorbar: bool = True,
    print_title: str = "Topographic Map with Shaded Relief",
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
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    vmin = -0.2 * max_elevation
    vmax = 1.1 * max_elevation
    im = ax.imshow(
        grid_z,
        cmap=topographic_cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
        alpha=1,
    )
    if print_colorbar:
        fig.colorbar(im, ax=ax, label="Altitude (m)")
    shaded = get_shading(grid_z, azimuth=45, altitude=315)
    plt.imshow(shaded, cmap="Greys", extent=extent, origin="lower", alpha=0.7)
    if print_title:
        plt.title("Topographic Map with Shaded Relief")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    return fig, ax


def get_shading(elevation: np.array, azimuth: float, altitude: float) -> np.array:
    """
    Get the shading of the elevation data.
    core code taken from: https://www.geophysique.be/2014/02/25/shaded-relief-map-in-python/

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


def plot_NP_map(
    np_name: str, figsize: tuple = (10, 10), relief_map_resolution: int = 100
):
    np_gdf = get_np_geo(np_name)
    np_bounds = tuple(np_gdf.total_bounds)
    # make the bounds square
    width = np_bounds[2] - np_bounds[0]
    height = np_bounds[3] - np_bounds[1]
    if width > height:
        missing_height = width - height
        np_bounds = (
            np_bounds[0],
            np_bounds[1] - missing_height / 2,
            np_bounds[2],
            np_bounds[3] + missing_height / 2,
        )
    else:
        missing_width = height - width
        np_bounds = (
            np_bounds[0] - missing_width / 2,
            np_bounds[1],
            np_bounds[2] + missing_width / 2,
            np_bounds[3],
        )
    outer_margin = 10000
    np_bounds_outer = (
        np_bounds[0] - outer_margin,
        np_bounds[1] - outer_margin,
        np_bounds[2] + outer_margin,
        np_bounds[3] + outer_margin,
    )
    inner_margin = 3000
    np_bounds_inner = (
        np_bounds[0] - inner_margin,
        np_bounds[1] - inner_margin,
        np_bounds[2] + inner_margin,
        np_bounds[3] + inner_margin,
    )
    cover_layer = "N50_Arealdekke_omrade"
    cover_gdf = gpd.read_file(N50_path, layer=cover_layer, bbox=np_bounds_outer)
    topographic_layer = "N50_Høyde_senterlinje"
    topographic_gdf = gpd.read_file(
        N50_path, layer=topographic_layer, bbox=np_bounds_outer
    )
    if relief_map_resolution == "auto":
        x_extent = np_bounds_outer[2] - np_bounds_outer[0]
        y_extent = np_bounds_outer[3] - np_bounds_outer[1]
        if x_extent > y_extent:
            grid_size_x = 5000
            grid_size_y = int(grid_size_x * y_extent / x_extent)
        else:
            grid_size_y = 5000
            grid_size_x = int(grid_size_y * x_extent / y_extent)
    else:
        grid_size_x = np.ceil(
            (np_bounds_outer[2] - np_bounds_outer[0]) / relief_map_resolution
        ).astype(int)
        grid_size_y = np.ceil(
            (np_bounds_outer[3] - np_bounds_outer[1]) / relief_map_resolution
        ).astype(int)
    if len(topographic_gdf) == 0:
        print(f"No topographic data (altitude lines) found for {np_name}")
        return
    #     print("No topographic data (altitude lines) found")
    #     cover_layer = "N50_Høyde_posisjon"
    #     topographic_gdf = gpd.read_file(
    #         N50_path, layer=cover_layer, bbox=np_bounds_outer
    #     )
    # else:
    grid_x, grid_y, grid_z, min_value, max_value = convert_multilinestring_to_mesh(
        topographic_gdf, grid_size=(grid_size_x, grid_size_y)
    )
    # if len(topographic_gdf) == 0:
    #     print("No topographic data for points found either - exiting")
    #     return
    # else:
    # points = np.array([(geom.x, geom.y) for geom in topographic_gdf.geometry])
    # values = topographic_gdf["hoyde"].values
    # grid_x, grid_y, grid_z = interpolate_topographic_data(
    #     points, values, grid_size=(grid_size_x, grid_size_y)
    # )

    # print(f"min and max elevation: {min_value}, {max_value} (based on the lines)")
    fig, ax = topographic_shaded_map(
        grid_x,
        grid_y,
        grid_z,
        figsize=figsize,
        print_colorbar=False,
        min_elevation=min_value,
        max_elevation=max_value,
        print_title=None,
    )
    fig, ax = plot_N50_cover(
        cover_gdf, fig=fig, ax=ax, plot_legend=False, plot_title=None
    )
    ax.set_xlim(np_bounds_inner[0], np_bounds_inner[2])
    ax.set_ylim(np_bounds_inner[1], np_bounds_inner[3])
    plt.axis("off")
    ax = plot_roads(np_bounds_inner, ax=ax)
    ax = plot_buildings(np_bounds_inner, ax=ax)
    ax = plot_protected_areas(np_name, np_bounds_inner, ax=ax)
    norway_boundary_gdf = get_Norway_boundaries()
    white_out_non_Norway(ax, norway_boundary_gdf, np_bounds_inner)
    ax = plot_location_in_Norway(np_gdf, norway_boundary_gdf, ax)
    np_gdf.plot(ax=ax, color="none", edgecolor="darkgreen", linewidth=6, linestyle="--")
    # ax.title.set_text(f"{np_name} National Park")

    if np_name == "Børgefjell/Byrkije":
        np_name = "Børgefjell_Byrkije"
    fig.savefig(
        root_dir / f"data/park_maps/{np_name}_map.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()
    return


# %%

if __name__ == "__main__":
    # print_national_park_names()
    NP_names = get_national_parks_names()
    # print(NP_names)
    test_NPs = [
        # "Jotunheimen",
        # "Sassen-Bünsow Land",
        # "Varangerhalvøya",
        # "Blåfjella-Skjækerfjella",
        # "Indre Wijdefjorden",
        # "Hardangervidda",
        # "Láhko",
        # "Van Mijenfjorden",
        # "Forlandet",
        # "Reinheimen",
        # "Nordvest-Spitsbergen",
        # "Rago",
        # "Stabbursdalen",
        # "Skarvan og Roltdalen",
        # "Øvre Pasvik",
        # "Sør-Spitsbergen",
        # "Anárjohka",
        # "Breheimen",
        # "Langsua",
        # "Dovre",
        # "Femundsmarka",
        "Forollhogna",
        #     "Jostedalsbreen",
        #     "Lomsdal-Visten",
        #     "Rondane",
        #     "Raet",
        #     "Seiland",
        #     "Reisa",
        #     "Junkerdal",
        #     "Lofotodden",
        #     "Sjunkhatten",
        #     "Dovrefjell-Sunndalsfjella",
        #     "Møysalen",
        #     "Ånderdalen",
        #     "Lierne",
        #     "Rohkunborri",
        #     "Folgefonna",
        #     "Færder",
        #     "Jomfruland",
        #     "Hallingskarvet",
        #     "Gutulia",
        #     "Nordre Isfjorden",
        #     "Øvre Dividal",
        #     "Ytre Hvaler",
        #     "Børgefjell/Byrkije",
        #     "Saltfjellet-Svartisen",
        #     "Fulufjellet",
        #     "Østmarka",
    ]
    for np_name in test_NPs:  # NP_names:
        plot_NP_map(np_name, figsize=(20, 10), relief_map_resolution="auto")

# %%
