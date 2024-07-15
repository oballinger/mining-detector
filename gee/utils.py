import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

from descarteslabs.geo import DLTile


def create_tiles(region, tilesize, padding, resolution):
    """
    Create a set of tiles that cover a region.
    Inputs:
        - region: a geojson polygon
        - tilesize: the size of the tiles in pixels
        - padding: the number of pixels to pad each tile
    Outputs:
        - tiles: a list of DLTile objects
    """
    tiles = DLTile.iter_from_shape(
        region, tilesize=tilesize, resolution=resolution, pad=padding
    )
    tiles = [tile for tile in tiles]

    return tiles


def pad_patch(patch, height, width=None):
    """
    Depending on how a polygon falls across pixel boundaries, the resulting patch can be slightly
    bigger or smaller than intended.
    pad_patch trims pixels extending beyond the desired number of pixels if the
    patch is larger than desired. If the patch is smaller, it will fill the
    edge by reflecting the values.
    If trimmed, the patch should be trimmed from the center of the patch.
    Inputs:
        - patch: a numpy array of the shape the model requires
        - height: the desired height of the patch
        - width (optional): the desired width of the patch
    Outputs:
        - padded_patch: a numpy array of the desired shape
    """
    if width is None:
        width = height

    patch_height, patch_width, _ = patch.shape

    if patch_height > height:
        trim_top = (patch_height - height) // 2
        trim_bottom = trim_top + height
    else:
        trim_top = 0
        trim_bottom = patch_height

    if patch_width > width:
        trim_left = (patch_width - width) // 2
        trim_right = trim_left + width
    else:
        trim_left = 0
        trim_right = patch_width

    trimmed_patch = patch[trim_top:trim_bottom, trim_left:trim_right, :]

    if patch_height < height:
        pad_top = (height - patch_height) // 2
        pad_bottom = pad_top + patch_height
        padded_patch = np.pad(
            trimmed_patch,
            ((pad_top, height - pad_bottom), (0, 0), (0, 0)),
            mode="reflect",
        )
    else:
        padded_patch = trimmed_patch

    if patch_width < width:
        pad_left = (width - patch_width) // 2
        pad_right = pad_left + patch_width
        padded_patch = np.pad(
            padded_patch,
            ((0, 0), (pad_left, width - pad_right), (0, 0)),
            mode="reflect",
        )

    return padded_patch


def chips_from_tile(data, tile, width, stride):
    """
    Break a larger tile of Sentinel data into a set of patches that
    a model can process.
    Inputs:
        - data: Sentinel data. Typically a numpy masked array
        - tile_coords: bounds of the tile in the format (west, south, east, north)
        - stride: number of pixels between each patch
    Outputs:
        - chips: A list of numpy arrays of the shape the model requires
        - chip_coords: A geodataframe of the polygons corresponding to each chip
    """
    (west, south, east, north) = tile.bounds
    delta_x = east - west
    delta_y = south - north
    x_per_pixel = delta_x / np.shape(data)[0]
    y_per_pixel = delta_y / np.shape(data)[1]

    # The tile is broken into the number of whole patches
    # Regions extending beyond will not be padded and processed
    chip_coords = []
    chips = []

    # Extract patches and create a shapely polygon for each patch
    for i in range(0, np.shape(data)[0] - width + stride, stride):
        for j in range(0, np.shape(data)[1] - width + stride, stride):
            patch = data[j : j + width, i : i + width]
            chips.append(patch)

            nw_coord = [west + i * x_per_pixel, north + j * y_per_pixel]
            ne_coord = [west + (i + width) * x_per_pixel, north + j * y_per_pixel]
            sw_coord = [west + i * x_per_pixel, north + (j + width) * y_per_pixel]
            se_coord = [
                west + (i + width) * x_per_pixel,
                north + (j + width) * y_per_pixel,
            ]
            tile_geometry = [nw_coord, sw_coord, se_coord, ne_coord, nw_coord]
            chip_coords.append(shapely.geometry.Polygon(tile_geometry))
    chip_coords = gpd.GeoDataFrame(geometry=chip_coords, crs=tile.crs)
    return chips, chip_coords

def minmax_norm(pixels):
    for i in range(pixels.shape[2]):
        pixels[:,:,i] = (pixels[:,:,i] - np.min(pixels[:,:,i])) / (np.max(pixels[:,:,i]) - np.min(pixels[:,:,i]))
    return pixels

def unit_norm(pixels, sensor):

    means = [
    {'sensor': 'S1', 'means': [-10.051072817856399, -16.651867464319356], 'stds': [2.1963779145197058, 2.2873515832032276]},
    {'sensor': 'NICFI', 'means': [693.2319128158244, 591.905562019799, 370.506283013815, 2193.8863101174647], 'stds': [566.4167817564434, 407.356998077997, 301.56748366624953, 832.109348018454]},
    {'sensor': 'S2', 'means': [1047.0421641825503, 820.0529029070626, 736.9112470910904, 657.3229316729462, 917.6294938035609, 1601.7804385065751, 1862.7748065344267, 2070.6779779384606, 1900.2215942671394, 1106.512413656176, 1527.2338024527187, 846.5729582225177], 'stds': [138.43361906215526, 242.0529108661862, 324.7921000512644, 461.1909194988024, 462.2156960074404, 590.9384137703064, 684.0157226477746, 767.9100518105192, 764.794394500607, 312.25277975004417, 755.6635139128948, 519.84684613692]},
    {'sensor': 'Multi', 'means': [-10.051072817856399, -16.651867464319356, 1047.0421641825503, 820.0529029070626, 736.9112470910904, 657.3229316729462, 917.6294938035609, 1601.7804385065751, 1862.7748065344267, 2070.6779779384606, 1900.2215942671394, 1106.512413656176, 1527.2338024527187, 846.5729582225177],'stds': [2.1963779145197058, 2.2873515832032276, 138.43361906215526, 242.0529108661862, 324.7921000512644, 461.1909194988024, 462.2156960074404, 590.9384137703064, 684.0157226477746, 767.9100518105192, 764.794394500607, 312.25277975004417, 755.6635139128948, 519.84684613692]}
    ]
    norms = [m for m in means if m['sensor'] == sensor]
    means=norms[0]['means']
    deviations=norms[0]['stds']    

    # normalize each band to global unit norm
    for i in range(pixels.shape[2]):
        pixels[:, :, i] = (pixels[:, :, i] - means[i]) / deviations[i]

    return pixels


def plot_numpy_grid(patches):
    num_img = int(np.ceil(np.sqrt(len(patches))))
    padding = 1
    h, w, c = patches[0].shape
    mosaic = np.zeros((num_img * (h + padding), num_img * (w + padding), c))
    counter = 0
    for i in range(num_img):
        for j in range(num_img):
            if counter < len(patches):
                mosaic[
                    i * (h + padding) : (i + 1) * h + i * padding,
                    j * (w + padding) : (j + 1) * w + j * padding,
                ] = patches[counter]
            else:
                mosaic[
                    i * (h + padding) : (i + 1) * h + i * padding,
                    j * (w + padding) : (j + 1) * w + j * padding,
                ] = np.zeros((h, w, c))
            counter += 1

    fig, ax = plt.subplots(figsize=(num_img, num_img), dpi=150)
    ax.axis("off")
    ax.imshow(mosaic)
    return fig
