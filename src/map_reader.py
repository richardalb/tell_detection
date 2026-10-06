"""
This module extracts the georeferenced locations of place-name labels from
scanned historical maps. Two interchangeable OCR backends are provided:

* :func:`paddle` uses PaddleOCR's PP-OCRv5 text-detection model.
* :func:`read_maps` uses EasyOCR.

Both slide a fixed-size window over the rasterised map, run OCR on each
window, convert the detected polygons to the map CRS and persist the results
as a CSV plus an ESRI shapefile. :func:`convert_to_coords` re-processes a saved
CSV into georeferenced polygons without rerunning OCR.

The map files are expected to have a georeferenced ``*_modified.tif``
companion produced during preprocessing.
"""

import easyocr
import rasterio as rio
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import glob
import ast
from shapely.geometry import Polygon
import geopandas as gpd
from paddleocr import TextDetection


def paddle(map_names, result_path):
    """Detect map text with PaddleOCR and write CSV / shapefile outputs.

    A ``1000`` px window with a ``250`` px step is slided over each map. Every
    detected polygon is translated into the raster's CRS using the affine
    transform and stored with its detection confidence.

    Args:
        map_names: Iterable of map file paths. Each is expected to have a
            ``*_modified.tif`` companion containing the georeferencing.
        result_path: Path of the CSV file to write; the shapefile is written
            alongside it with the extension replaced by ``.shp``.
    """
    model = TextDetection(model_name="PP-OCRv5_server_det")
    window_size = 1000
    step_size = window_size // 4
    results = []
    for map_name in map_names:
        image = np.array(Image.open(map_name[:-4] + "_modified.tif"))
        with rio.open(map_name[:-4] + "_modified.tif") as src:
            crs = src.crs
            transform = src.transform

        map = Image.open(map_name[:-4] + "_modified.tif")
        map_height, map_width = map.size

        for i in tqdm(range(0, int((map_width-1000) // step_size))):
            for j in range(0, int((map_height-1000) // step_size)):
                output = model.predict(
                    image[i*step_size:(i*step_size) + window_size,
                          j*step_size:(j*step_size) + window_size, :],
                    batch_size=1,
                )
                for o in output:
                    for text, confidence in zip(o['dt_polys'], o['dt_scores']):
                        coords = []
                        for corner in text:
                            # Shift the window-local corner into global pixel
                            # space and then into map coordinates.
                            coords.append(
                                transform *
                                (int(corner[0] + (j * step_size)),
                                 int(corner[1] + (i * step_size)))
                            )
                        coords = Polygon(coords)
                        results.append(
                            {'bounding_box': text, 'probability': confidence, 'map': map_name, 'polygon': coords})

    result_df = pd.DataFrame(results)
    result_geodf = gpd.GeoDataFrame(
        result_df, geometry=result_df['polygon'], crs=crs)
    result_df.to_csv(result_path)
    result_geodf.to_file(result_path[:-4] + '.shp')


def convert_to_coords(data_path):
    """Convert an OCR result CSV into georeferenced polygons.

    Reads the CSV produced by :func:`paddle` / :func:`read_maps`, filters out
    detections below a ``0.3`` confidence threshold, converts every bounding
    box to a polygon in the map CRS and writes a shapefile suffixed with
    ``_coords``.

    Args:
        data_path: Path to the OCR CSV with a ``bounding_box`` column.
    """
    # converts bounding boxes into polygons with coordinates
    data = pd.read_csv(data_path)

    # The CRS is taken from the first map referenced in the table.
    with rio.open(data.iloc[0]['map']) as src:
        crs = src.crs

    data = data[data['probability'] > 0.3]
    data['polygons'] = data.apply(convert, axis=1)
    data = gpd.GeoDataFrame(data, geometry=data['polygons'], crs=crs)
    data.to_file(data_path.split('.')[0] + '_coords.shp')


def convert(x):
    """Convert a single CSV row's serialised bounding box into a polygon.

    The ``bounding_box`` field is stored as a stringified NumPy literal; the
    NumPy type wrappers are stripped before parsing with :func:`ast.literal_eval`.
    """
    map_path = x['map']
    with rio.open(map_path) as src:
        transform = src.transform
        print(src.crs)

    # Remove the ``np.int32(...)`` / ``np.float64(...)`` wrappers so that the
    # remaining text is a valid Python literal.
    bboxes = x['bounding_box'].replace('np.int32(', '')
    bboxes = bboxes.replace('np.float64(', '')
    bboxes = bboxes.replace(')', '')
    bboxes = ast.literal_eval(bboxes)
    corners = []
    for bbox in bboxes:
        lat, lon = transform * bbox
        corners.append((lat, lon))

    result = Polygon(corners)
    return result


def read_maps(map_names, result_path):
    """Detect map text with EasyOCR and write CSV / shapefile outputs.

    Uses a ``384`` px window (matching the EasyOCR model) with a ``96`` px step.
    Each window is cropped from the PIL image and passed to EasyOCR; detections
    are shifted to global pixel coordinates, transformed to the map CRS and
    saved as both CSV and shapefile.

    Args:
        map_names: Iterable of map file paths, each with a ``*_modified.tif``
            georeferenced companion.
        result_path: Path of the CSV file to write; the shapefile is written
            alongside it with the extension replaced by ``.shp``.
    """
    window_size = 384  # based of https://huggingface.co/qualcomm/EasyOCR
    step_size = int(384 / 4)

    reader = easyocr.Reader(['en'])
    results = []

    for map_name in map_names:
        with rio.open(map_name[:-4] + "_modified.tif") as src:
            crs = src.crs
            transform = src.transform
            print(map_name, crs, transform)

        map = Image.open(map_name)
        map_width, map_height = map.size

        for i in tqdm(range(0, int(map_width // step_size))):
            for j in range(0, int(map_height // step_size)):
                cutout = map.crop(
                    (i * step_size, j*step_size, (i*step_size)+window_size, (j*step_size)+window_size))
                result = reader.readtext(np.asarray(cutout))

                for text in result:
                    coords = []
                    for corner in text[0]:
                        # Shift the window-local corner into global pixel space
                        # and then into map coordinates.
                        coords.append(
                            transform * (int(corner[0] + (i * step_size)), int(corner[1] + (j * step_size))))

                    coords = Polygon(coords)

                    results.append(
                        {'bounding_box': text[0], 'text': text[1], 'probability': text[2], 'map': map_name, 'polygon': coords})

    result_df = pd.DataFrame(results)
    result_geodf = gpd.GeoDataFrame(
        result_df, geometry=result_df['polygon'], crs=crs)
    result_df.to_csv(result_path)
    result_geodf.to_file(result_path[:-4] + '.shp')


if __name__ == '__main__':
    # Batch process every historical map found under ``data/maps``.
    map_names = glob.glob('data/maps/*.jpg')
    print(map_names)
    result_path = 'historic_map_locations.csv'
    paddle(map_names, result_path)
