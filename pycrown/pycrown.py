"""
PyCrown - Fast raster-based individual tree segmentation for LiDAR data
-----------------------------------------------------------------------
Copyright: 2018, Jan Zörner
Licence: GNU GPLv3
"""

import time
import platform
import warnings
from math import floor
from pathlib import Path

import pyximport

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial.distance import cdist

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
# from skimage.feature import peak_local_max
from scipy.ndimage import maximum_filter, label
from osgeo import gdal
from osgeo import osr

from shapely.geometry import mapping, Point, Polygon, GeometryCollection, LineString, Point

from rasterio.features import shapes as rioshapes
from rasterio.mask import mask

import fiona
from fiona.crs import from_epsg

import laspy
from matplotlib import pyplot as plt

try:
    from pycrown import _crown_dalponte_cython
except ImportError:
    print("WARNING: Cython module not compiled. 'crown_dalponte_cython' not available")
from pycrown import _crown_dalponte_numba
from pycrown import _crown_dalponteCIRC_numba

gdal.UseExceptions()
warnings.filterwarnings('ignore')


class NoTreesException(Exception):
    """ Raised when no tree detected """
    pass


class GDALFileNotFoundException(Exception):
    """ Raised when GDAL file not found """
    pass


class PyCrown:
    __author__ = "Dr. Jan Zörner"
    __copyright__ = "Copyright 2018, Jan Zörner"
    __credits__ = ["Jan Zörner", "John Dymond", "James Shepherd", "Ben Jolly"]
    __license__ = "GNU GPLv3"
    __version__ = "0.2"
    __maintainer__ = "Jan Zörner"
    __email__ = "zoernerj@landcareresearch.co.nz"
    __status__ = "Development"
    __modifiedby__ = "Dr Jonathan Monsalve 2025"

    def __init__(self, chm_file, dtm_file, dsm_file, las_file=None,
                 outpath=None, suffix=None, vegetation_maks_path=None, resolution=None):
        """ PyCrown class

        Parameters
        ----------
        chm_file :  str
                    Path to Canopy Height Model
        dtm_file :  str
                    Path to Digital Terrain Model
        dsm_file :  str
                    Path to Digital Surface Model
        las_file :  str
                    Path to LAS (LiDAR point cloud) file
        outpath  :  str, optional
                    Output directory
        suffix   :  str, optional
                    text appended to output file names

        Example
        -------

        PC = PyCrown(F_CHM, F_DTM, F_DSM, F_LAS, outpath=sys.argv[1])
        PC.filter_chm(5)
        PC.tree_detection(PC.chm, ws=5, hmin=16.)
        PC.crown_delineation(algorithm='dalponteCIRC_numba', th_tree=15.,
                             th_seed=0.7, th_crown=0.55, max_crown=10.)
        PC.correct_tree_tops()
        PC.get_tree_height_elevation(loc='top')
        PC.get_tree_height_elevation(loc='top_cor')
        PC.screen_small_trees(hmin=20., loc='top')
        PC.crowns_to_polys_raster()
        PC.crowns_to_polys_smooth(store_las=True)
        PC.quality_control()
        PC.export_raster(PC.chm, PC.outpath / 'chm.tif', 'CHM')
        PC.export_tree_locations(loc='top')
        PC.export_tree_locations(loc='top_cor')
        PC.export_tree_crowns(crowntype='crown_poly_raster')
        PC.export_tree_crowns(crowntype='crown_poly_smooth')
        """

        suffix = f'_{suffix}' if suffix else ''

        self.outpath = Path(outpath) if outpath else Path('./')
        self.vegetation_maks_path = vegetation_maks_path
        # Load the CHM
        self.chm_file = Path(chm_file)
        try:
            chm_gdal = gdal.Open(str(self.chm_file), gdal.GA_ReadOnly)
        except RuntimeError as e:
            raise IOError(e)
        proj = osr.SpatialReference(wkt=chm_gdal.GetProjection())
        self.epsg = int(proj.GetAttrValue('AUTHORITY', 1))
        self.srs = from_epsg(self.epsg)
        self.geotransform = chm_gdal.GetGeoTransform()
        if resolution is None:
            self.resolution = abs(self.geotransform[-1])
        else:
            self.resolution = resolution
        print("resolution", self.resolution)
        self.ul_lon = chm_gdal.GetGeoTransform()[0]
        self.ul_lat = chm_gdal.GetGeoTransform()[3]
        try:
            self.chm0 = chm_gdal.GetRasterBand(1).ReadAsArray()
        except:
            with rasterio.open(self.chm_file) as src:
                # Leer el raster como un array de NumPy
                self.chm0 = src.read(1)
        chm_gdal = None

        # Load the DTM
        try:
            self.dtm_file = Path(dtm_file)
        except RuntimeError as e:
            raise IOError(e)
        dtm_gdal = gdal.Open(str(self.dtm_file), gdal.GA_ReadOnly)
        try:
            self.dtm = dtm_gdal.GetRasterBand(1).ReadAsArray()
        except:
            with rasterio.open(self.dtm_file) as src:
                # Leer el raster como un array de NumPy
                self.dtm = src.read(1)
        dtm_gdal = None

        # Load the DSM
        try:
            self.dsm_file = Path(dsm_file)
        except RuntimeError as e:
            raise IOError(e)

        # self.dsm = dsm_gdal.GetRasterBand(1).ReadAsArray()
        try:
            dsm_gdal = gdal.Open(str(self.dsm_file), gdal.GA_ReadOnly)
            self.dsm = dsm_gdal.GetRasterBand(1).ReadAsArray()
        except:
            with rasterio.open(self.dsm_file) as src:
                # Leer el raster como un array de NumPy
                self.dsm = src.read(1)
        dsm_gdal = None

        # Load the LiDAR point cloud
        self.lidar_in_crowns = None
        self.las = None
        if las_file:
            self._load_lidar_points_cloud(las_file)

        self.chm = None
        self.crowns = None
        self.tree_markers = None
        self.tt_corrected = None

        self.define_trees_df()

    def define_trees_df(self):
        self.trees = pd.DataFrame(columns=[
            'top_height', 'top_elevation',
            'top_cor_height', 'top_cor_elevation',
            'crown_poly_raster', 'crown_poly_smooth',
            'top_cor', 'top', 'tt_corrected'
        ])

        self.trees = self.trees.astype(dtype={
            'top_height': 'float',
            'top_elevation': 'float',
            'top_cor_height': 'float',
            'top_cor_elevation': 'float',
            'crown_poly_raster': 'object',
            'crown_poly_smooth': 'object',
            'top_cor': 'object',
            'top': 'object',
            'tt_corrected': 'int'
        })

    def _load_lidar_points_cloud(self, fname):
        """ Loads LiDAR dataset

        Parameters
        ----------
        fname :   str
                  Path to LiDAR dataset (.las or .laz-file)
        """
        with laspy.open(str(fname)) as las:
            # Leer los puntos LIDAR
            points = las.read()
            # Convertir los datos a un array de numpy
            lidar_points = np.array((
                points.x, points.y, points.z, points.intensity, points.return_number,
                points.classification
            )).transpose()
            # Crear un DataFrame de pandas
            colnames = ['x', 'y', 'z', 'intensity', 'return_num', 'classification']
            self.las = pd.DataFrame(lidar_points, columns=colnames)

    def _check_empty(self):
        """ Helper function raising an Exception if no trees present

        Raises
        ------
        NoTreesException
            raises Exception if no trees present
        """
        if self.trees.empty:
            raise NoTreesException

    def _to_lonlat(self, pix_x, pix_y, resolution):
        ''' Convert pixel coordinates to longitude/latitude

        Parameters
        ----------
        pix_x :      int, float, ndarray
                     Column coordinate of raster
        pix_y :      int, float, ndarray
                     Row coordinate of raster
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        tuple
            longitude(s), latitude(s)
        '''
        lon = self.ul_lon + (pix_x * resolution)
        lat = self.ul_lat - (pix_y * resolution)
        return lon, lat

    def _to_colrow(self, lon, lat, resolution):
        ''' Convert longitude/latitude to pixel coordinates
        returns either tuple of floats or 2xn ndarray

        Parameters
        ----------
        lon :        int, float, ndarray, (pandas) Series
                     Longtitude
        lat :        int, float, ndarray, (pandas) Series
                     Latitude
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        tuple
            Column/Row coordinate as floats
        or:
        ndarray
            Column/Row coordinate as 2xn ndarray
        '''
        x = (lon - self.ul_lon) / resolution
        y = (self.ul_lat - lat) / resolution
        if isinstance(x, type(y)):
            if isinstance(x, float):
                return int(x), int(y)
            if isinstance(x, (np.ndarray, pd.Series)):
                return np.array([x, y], dtype=int)
        else:
            raise Exception("Can't handle different input types for x, y.")

    def _get_z(self, lon, lat, band, resolution):
        """ Returns data from raster band for coordinate location(s)

        Parameters
        ----------
        lon :        int, float, ndarray, (pandas) Series
                     Longtitude
        lat :        int, float, ndarray, (pandas) Series
                     Latitude
        band :       ndarray
                     raster layer (e.g., CHM or DSM)
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        float
            raster value at longitude/latitude position
        """
        x, y = self._to_colrow(lon, lat, resolution)
        return band[y, x]

    def _tree_lonlat(self, loc='top'):
        ''' returns longitude/latitude of tree tops

        Parameters
        ----------
        loc :    str, optional
                 initial or corrected tree top location: `top` or `top_cor`

        Returns
        -------
        tuple
            ndarrays of longitude(s), latitude(s) of tree tops
        '''
        lons = np.array([tree[1][loc].x for tree in self.trees.iterrows()])
        lats = np.array([tree[1][loc].y for tree in self.trees.iterrows()])
        return lons, lats

    def _tree_colrow(self, loc, resolution):
        """ returns column/row of tree tops

        Parameters
        ----------
        loc :        str, optional
                     initial or corrected tree top location: `top` or `top_cor`
        resolution:  int
                     resolution (in m) of raster

        Returns
        -------
        ndarray
            2xn ndarray of column(s), row(s) positions of tree tops
        """
        return self._to_colrow(np.array([tree.x for tree in self.trees[loc]]),
                               np.array([tree.y for tree in self.trees[loc]]),
                               resolution).astype(np.int32)

    def _watershed(self, inraster, th_tree=15.):
        """ Simple implementation of a watershed tree crown delineation

        Parameters
        ----------
        inraster :   ndarray
                     raster of height values (e.g., CHM)
        th_tree :    float
                     minimum height of tree crown

        Returns
        -------
        ndarray
            raster of individual tree crowns
        """

        inraster_mask = inraster.copy()
        inraster_mask[inraster <= th_tree] = 0

        if self.vegetation_maks_path is not None:
            with rasterio.open(self.vegetation_maks_path) as src:
                vegetation_mask = src.read(1)  # Read the first band
                vegetation_meta = src.meta.copy()
            if vegetation_mask.shape != inraster_mask.shape:
                raise ValueError("Shapes of vegetation mask and inraster_mask do not match.")
            # Combine the masks using logical AND
            inraster_mask = (inraster_mask) * ((vegetation_mask > 0) * 1.0)
        raster = inraster.copy()
        raster[np.isnan(raster)] = 0.
        labels = watershed(-raster, self.tree_markers, mask=combined_mask)
        return labels

    def _slic(self, inraster, th_tree=15., n_segments=300):
        """ Simple implementation of a slic tree crown delineation

        Parameters
        ----------
        inraster :   ndarray
                     raster of height values (e.g., CHM)
        th_tree :    float
                     minimum height of tree crown

        Returns
        -------
        ndarray
            raster of individual tree crowns
        """
        inraster_mask = inraster.copy()
        inraster_mask[inraster <= th_tree] = 0
        if self.vegetation_maks_path is not None:
            with rasterio.open(self.vegetation_maks_path) as src:
                vegetation_mask = src.read(1)  # Read the first band
                vegetation_meta = src.meta.copy()
            if vegetation_mask.shape != inraster_mask.shape:
                raise ValueError("Shapes of vegetation mask and inraster_mask do not match.")
            # Combine the masks using logical AND
            inraster_mask = (inraster_mask) * ((vegetation_mask > 0) * 1.0)
        raster = inraster.copy()
        raster[np.isnan(raster)] = 0.
        labels = slic(raster, n_segments=n_segments, compactness=10, mask=inraster_mask, channel_axis=None)
        return labels

    def _felzenszwalb(self, inraster, th_tree=15.):
        """ Simple implementation of a felzenszwalb tree crown delineation

        Parameters
        ----------
        inraster :   ndarray
                     raster of height values (e.g., CHM)
        th_tree :    float
                     minimum height of tree crown

        Returns
        -------
        ndarray
            raster of individual tree crowns
        """
        inraster_mask = inraster.copy()
        inraster_mask[inraster <= th_tree] = 0
        raster = inraster.copy()
        raster[np.isnan(raster)] = 0.
        raster1 = raster * ((inraster_mask > 0) * 1.0)
        labels = quickshift(raster1, kernel_size=3, max_dist=6, ratio=0.5, channel_axis=None)
        plt.imshow(labels)
        plt.show()
        return labels

    def _screen_crowns(self, cond):
        """ Remove crowns outside tile from crowns raster and reindex
        the remaining ones

        Parameters
        ----------
        cond :    list
                  list of booleans. Keep trees/crowns with True
        """
        counter = 1
        for idx, valid in enumerate(cond):
            if valid:
                self.crowns[self.crowns == idx + 1] = counter
                counter += 1
            else:
                self.crowns[self.crowns == idx + 1] = 0.

    @staticmethod
    def _get_kernel(radius=5, circular=False):
        """ returns a block or disc-shaped filter kernel with given radius

        Parameters
        ----------
        radius :    int, optional
                    radius of the filter kernel
        circular :  bool, optional
                    set to True for disc-shaped filter kernel, block otherwise

        Returns
        -------
        ndarray
            filter kernel
        """
        if circular:
            y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            return x ** 2 + y ** 2 <= radius ** 2
        else:
            return np.ones((int(radius), int(radius)))

    def _smooth_raster(self, raster, ws, resolution, circular=False):
        """ Smooth a raster with a median filter

        Parameters
        ----------
        raster :      ndarray
                      raster to be smoothed
        ws :          int
                      window size of smoothing filter
        resolution :  int
                      resolution of raster in m
        circular :    bool, optional
                      set to True for disc-shaped filter kernel, block otherwise

        Returns
        -------

        ndarray
            smoothed raster
        """
        return filters.median_filter(
            raster, footprint=self._get_kernel(ws, circular=circular))

    def clip_data_to_bbox(self, bbox, las_offset=10):
        """ Clip input data to subset region based on bounding box

        Parameters
        ----------
        bbox :    tuple
                  lon_min, lon_max, lat_min, lat_max
        las_offset :  int, optional
                      buffer around bounding for LiDAR data (in m)
        """

        lon_min, lon_max, lat_min, lat_max = bbox
        col_min, row_max = self._to_colrow(lon_min, lat_min, self.resolution)
        col_max, row_min = self._to_colrow(lon_max, lat_max, self.resolution)

        self.chm0 = self.chm0[row_min:row_max, col_min:col_max]
        if isinstance(self.chm, np.ndarray):
            self.chm = self.chm[row_min:row_max, col_min:col_max]
        self.dtm = self.dtm[row_min:row_max, col_min:col_max]
        self.dsm = self.dsm[row_min:row_max, col_min:col_max]
        lasmask = (
                (self.las.x >= lon_min - las_offset) &
                (self.las.x <= lon_max + las_offset) &
                (self.las.y >= lat_min - las_offset) &
                (self.las.y <= lat_max + las_offset)
        )
        self.las = self.las[lasmask]

        self.ul_lon = lon_min
        self.ul_lat = lat_max

    def get_tree_height_elevation(self, loc='top'):
        ''' Sets tree height and elevation in tree dataframe

        Parameters
        ----------
        loc :    str, optional
                 initial or corrected tree top location: `top` or `top_cor`
        '''
        lons, lats = self._tree_lonlat(loc)
        self.trees[f'{loc}_height'] = self._get_z(
            lons, lats, self.chm, self.resolution)
        self.trees[f'{loc}_elevation'] = self._get_z(
            lons, lats, self.dtm, self.resolution)

    def filter_chm(self, ws, ws_in_pixels=False, circular=False):
        ''' Pre-process the canopy height model (smoothing and outlier removal).
        The original CHM (self.chm0) is not overwritten, but a new one is
        stored (self.chm).

        Parameters
        ----------
        ws :            int
                        window size of smoothing filter in metre (set in_pixel=True, otherwise)
        ws_in_pixels :  bool, optional
                        sets ws in pixel
        circular :      bool, optional
                        set to True for disc-shaped filter kernel, block otherwise
        '''
        if not ws_in_pixels:
            if ws % self.resolution:
                raise Exception(
                    "Image filter size not an integer number. Please check if image resolution matches filter size (in metre or pixel).")
            else:
                ws = int(ws / self.resolution)

        self.chm = self._smooth_raster(self.chm0, ws, self.resolution,
                                       circular=circular)
        self.chm0[np.isnan(self.chm0)] = 0.
        zmask = (self.chm < 0.5) | np.isnan(self.chm) | (self.chm > 60.)
        self.chm[zmask] = 0

    def tree_detection(self, raster, resolution=None, ws=5, hmin=20,
                       return_trees=False, ws_in_pixels=False):
        ''' Detect individual trees from CHM raster based on a maximum filter.
        Identified trees are either stores as list in the tree dataframe or
        returned as ndarray.

        Parameters
        ----------
        raster :        ndarray
                        raster of height values (e.g., CHM)
        resolution :    int, optional
                        resolution of raster in m
        ws :            float
                        moving window size (in metre) to detect the local maxima
        hmin :          float
                        Minimum height of a tree. Threshold below which a pixel
                        or a point cannot be a local maxima
        return_trees :  bool
                        set to True if detected trees shopuld be returned as
                        ndarray instead of being stored in tree dataframe
        ws_in_pixels :  bool
                        sets ws in pixel

        Returns
        -------
        ndarray (optional)
            nx2 array of tree top pixel coordinates
        '''

        if not isinstance(raster, np.ndarray):
            raise Exception("Please provide an input raster as numpy ndarray.")

        resolution = resolution if resolution else self.resolution

        if not ws_in_pixels:
            if ws % resolution:
                raise Exception(
                    "Image filter size not an integer number. Please check if image resolution matches filter size (in metre or pixel).")
            else:
                ws = int(ws / resolution)

        # Maximum filter to find local peaks
        raster_maximum = filters.maximum_filter(
            raster, footprint=self._get_kernel(ws, circular=True))
        tree_maxima = raster == raster_maximum

        # alternative using skimage peak_local_max
        # chm = inraster.copy()
        # chm[np.isnan(chm)] = 0.
        # tree_maxima = peak_local_max(chm, indices=False, footprint=kernel)

        # remove tree tops lower than minimum height
        tree_maxima[raster <= hmin] = 0
        # distance = ndimage.distance_transform_edt(tree_maxima)
        # separated_maxima = (distance == filters.maximum_filter(distance, size=3))
        # label each tree
        self.tree_markers, num_objects = ndimage.label(tree_maxima)

        if num_objects == 0:
            raise NoTreesException

        # if canopy height is the same for multiple pixels,
        # place the tree top in the center of mass of the pixel bounds
        yx = np.array(
            ndimage.center_of_mass(
                raster, self.tree_markers, range(1, num_objects + 1)
            ), dtype=np.float32
        ) + 0.5
        xy = np.array((yx[:, 1], yx[:, 0])).T

        trees = [Point(*self._to_lonlat(xy[tidx, 0], xy[tidx, 1], resolution))
                 for tidx in range(len(xy))]

        if return_trees:
            return np.array(trees, dtype=object), xy
        else:
            df = pd.DataFrame(np.array([trees, trees], dtype='object').T,
                              dtype='object', columns=['top_cor', 'top'])

            self.trees = pd.concat([self.trees, df], ignore_index=True)
            print("len tree_detection", len(self.trees))

        self._check_empty()

    def reassign_disconnected_labels(self, crowns):
        # Paso 1: Crear una copia del array de crowns para modificarlo sin alterar el original
        labeled_array = np.copy(crowns)

        # Paso 2: Inicializar el máximo de etiquetas actuales en crowns
        max_label = np.max(crowns)

        # Paso 3: Procesar cada label individualmente
        for current_label in np.unique(crowns):
            if current_label == 0:  # Ignorar el fondo (etiqueta 0)
                continue

            # Paso 3.1: Extraer la región que corresponde al label actual
            region_mask = (crowns == current_label)

            # Paso 3.2: Etiquetar los componentes conectados dentro de la región actual
            labeled_region = label(region_mask)
            # print(labeled_region[0])
            # Paso 3.3: Obtener las propiedades de las regiones conectadas
            cont = 0
            for region in regionprops(labeled_region[0]):
                # Extraer las coordenadas de la región conectada
                cont += 1
                if cont == 1:
                    continue
                region_coords = region.coords

                # Asignar un nuevo label único a las regiones desconectadas
                new_label = max_label + 1  # Asignar un nuevo label
                max_label += 1  # Incrementar el máximo para evitar repetir etiquetas

                # Asignar el nuevo label a los píxeles de esta región desconectada
                for coord in region_coords:
                    y, x = coord
                    labeled_array[y, x] = new_label

        # Paso 4: Devolver el arreglo con los labels reasignados
        return labeled_array

    def redefinition_trees(self, raster, return_trees=False):
        # if canopy height is the same for multiple pixels,
        # place the tree top in the center of mass of the pixel bounds
        num_objects = self.tree_markers.max()

        yx = np.array(
            ndimage.center_of_mass(
                raster, self.tree_markers, range(1, num_objects + 1)
            ), dtype=np.float32
        ) + 0.5

        xy = np.array((yx[:, 1], yx[:, 0])).T

        trees = [Point(*self._to_lonlat(xy[tidx, 0], xy[tidx, 1], self.resolution))
                 for tidx in range(len(xy))]
        self.define_trees_df()
        if return_trees:
            return np.array(trees, dtype=object), xy
        else:
            df = pd.DataFrame(np.array([trees, trees], dtype='object').T,
                              dtype='object', columns=['top_cor', 'top'])

            self.trees = pd.concat([self.trees, df], ignore_index=True)
            print("redefine tree_detection count ", len(self.trees))

        self._check_empty()

    def region_size_filter(self, min_size):
        """
        Generates a binary vector indicating whether each region in a labeled image
        is larger than a given pixel threshold. and then remove it

        Parameters:
            labeled_image (numpy.ndarray): A 2D array where each unique positive integer
                                        represents a region label (1 to N).
            min_size (int): The minimum size (in pixels) a region must have to be marked as valid.

        Returns:
            numpy.ndarray: A binary vector of size N (number of regions), where each entry is:
                        - 1 if the corresponding region is larger than `min_size`.
                        - 0 otherwise.
        """
        # Get the unique region labels and their sizes
        region_labels, region_sizes = np.unique(self.crowns, return_counts=True)

        # Exclude the background (label 0) if present
        if region_labels[0] == 0:
            region_labels = region_labels[1:]
            region_sizes = region_sizes[1:]

        # Generate the binary vector
        binary_vector = (region_sizes > min_size).astype(int)
        print("bin ", len(binary_vector))
        print("crowns ", self.crowns.max())
        if isinstance(self.crowns, np.ndarray):
            self._screen_crowns(binary_vector)
        self.tree_markers = self.crowns
        self.redefinition_trees(self.crowns)
        print("crowns 2 ", self.crowns.max())

    def crown_delineation(self, algorithm, loc='top', **kwargs):
        """ Function calling external crown delineation algorithms

        Parameters
        ----------
        algorithm :  str
                     crown delineation algorithm to be used, choose from:
                     ['dalponte_cython', 'dalponte_numba',
                      'dalponteCIRC_numba', 'watershed_skimage', 'slic_skimage', 'felzenszwalb_skimage']
        loc :        str, optional
                     tree seed position: `top` or `top_cor`
        th_seed :    float
                     factor 1 for minimum height of tree crown
        th_crown :   float
                     factor 2 for minimum height of tree crown
        th_tree :    float
                     minimum height of tree seed (in m)
        max_crown :  float
                     maximum radius of tree crown (in m)
        n_segments: int
                    number of segmets for algorithm slic (all others but th_tree are ignored)

        Returns
        -------
        ndarray
            raster of individual tree crowns
        """
        timeit = 'Tree crowns delineation: {:.3f}s'

        # get the tree seeds (starting points for crown delineation)
        seeds = self._tree_colrow(loc, self.resolution)
        inraster = kwargs.get('inraster')

        if not isinstance(inraster, np.ndarray):
            inraster = self.chm
        else:
            inraster = inraster

        if kwargs.get('max_crown'):
            max_crown = kwargs['max_crown'] / self.resolution

        if algorithm == 'dalponte_cython':
            tt = time.time()
            crowns = _crown_dalponte_cython._crown_dalponte(
                inraster, seeds,
                th_seed=float(kwargs['th_seed']),
                th_crown=float(kwargs['th_crown']),
                th_tree=float(kwargs['th_tree']),
                max_crown=float(max_crown)
            )
            print(timeit.format(time.time() - tt))

        elif algorithm == 'dalponte_numba':
            tt = time.time()
            crowns = _crown_dalponte_numba._crown_dalponte(
                inraster, seeds,
                th_seed=float(kwargs['th_seed']),
                th_crown=float(kwargs['th_crown']),
                th_tree=float(kwargs['th_tree']),
                max_crown=float(max_crown)
            )
            print(timeit.format(time.time() - tt))

        elif algorithm == 'dalponteCIRC_numba':
            tt = time.time()
            crowns = _crown_dalponteCIRC_numba._crown_dalponteCIRC(
                inraster, seeds,
                th_seed=float(kwargs['th_seed']),
                th_crown=float(kwargs['th_crown']),
                th_tree=float(kwargs['th_tree']),
                max_crown=float(max_crown)
            )
            print(timeit.format(time.time() - tt))

        elif algorithm == 'watershed_skimage':
            tt = time.time()
            crowns = self._watershed(
                inraster, th_tree=float(kwargs['th_tree'])
            )
            print(timeit.format(time.time() - tt))
        elif algorithm == 'slic_skimage':
            tt = time.time()
            crowns = self._slic(
                inraster, th_tree=float(kwargs['th_tree']), n_segments=int(kwargs['n_segments'])
            )
            crowns = self.reassign_disconnected_labels(crowns)
            self.tree_markers = crowns
            self.redefinition_trees(inraster)
            print(timeit.format(time.time() - tt))
        elif algorithm == 'felzenszwalb_skimage':
            tt = time.time()
            crowns = self._felzenszwalb(
                inraster, th_tree=float(kwargs['th_tree'])
            )
            crowns = self.reassign_disconnected_labels(crowns)
            self.tree_markers = crowns
            self.redefinition_trees(inraster)
            print(timeit.format(time.time() - tt))
        self.crowns = np.array(crowns, dtype=np.int32)
        # arbol más grande que 0.5m2
        pixels = int(0.5 / self.resolution ** 2)
        self.region_size_filter(pixels)
        if isinstance(self.crowns, np.ndarray) and algorithm != 'slic_skimage':
            self._screen_crowns(self.condr)

    def clip_trees_to_bbox(self, bbox=None, inbuf=None, f_tiles=None, row=None,
                           col=None, loc='top'):
        """ Clip tree tops and crowns to bounding box or tile extent.
        Tree dataframe is updated with subset of trees.

        Parameters
        ----------
        bbox :     tuple, optional
                   floats for (lon_min, lon_max, lat_min, lat_max)
        inbuf :    integer or float, optional
                   distance of inward buffer in metres
        f_tiles :  str, optional
                   Path to LiDAR tiles polygon with coordinates for all
                   bounding boxes for each tile
        row :      int, optional
                   row number of tile
        col :      int, optional
                   column number of tile
        loc :      str, optional
                   tree seed position: `top` or `top_cor`
        """
        if bbox:
            lon_min, lon_max, lat_min, lat_max = bbox
        elif inbuf:
            lat_max = self.ul_lat - inbuf
            lat_min = self.ul_lat - (self.chm.shape[0] * self.resolution) - inbuf
            lon_min = self.ul_lon + inbuf
            lon_max = self.ul_lon + (self.chm.shape[1] * self.resolution) - inbuf
        elif f_tiles:
            # get the bounding box of each tile
            with fiona.open(f_tiles, 'r') as tilepolys_layer:
                tiles = {}
                for tile in tilepolys_layer:
                    r = tile['properties']['Row']
                    c = tile['properties']['Col']
                    lon_min = tile['geometry']['coordinates'][0][0][0]
                    lat_min = tile['geometry']['coordinates'][0][0][1]
                    lon_max = tile['geometry']['coordinates'][0][2][0]
                    lat_max = tile['geometry']['coordinates'][0][1][1]
                    tiles[r, c] = lon_min, lat_min, lon_max, lat_max

            # identify and clip tree tops inside tile
            lon_min, lat_min, lon_max, lat_max = tiles[row, col]
        else:
            raise Exception("No clipping method specified.")

        tree_lons, tree_lats = self._tree_lonlat(loc)
        cond = (
                (tree_lons >= lon_min) & (tree_lons < lon_max) &
                (tree_lats >= lat_min) & (tree_lats < lat_max)
        )
        self.trees = self.trees[cond]
        self.condr = cond
        self.trees = self.trees.reset_index(drop=True)
        if isinstance(self.crowns, np.ndarray):
            self._screen_crowns(cond)

    def correct_tree_tops(self, check_all=False):
        """ Correct the location of tree tops in steep terrain.
        Tree dataframe is updated with corrected tree top positions (`top_cor`).

        Parameters
        ----------
        check_all :    bool, optional
                       set to True if all trees should be corrected, ignoring
                       whether they are located on steep terrain
        """

        print(f'Number of trees: {len(self.trees)}')

        # calculate center of mass of crowns
        comass = np.array(
            ndimage.center_of_mass(self.crowns, self.crowns,
                                   range(1, self.crowns.max() + 1))
        )

        corr_n = 0
        corr_dsm = 0
        corr_com = 0

        for tidx in range(len(self.trees)):
            tree = self.trees.iloc[tidx]
            col, row = self._to_colrow(tree['top'].x, tree['top'].y,
                                       self.resolution)
            rcindices = np.where(self.crowns == tidx + 1)

            dtm_mean = np.nanmean(self.dtm[rcindices])
            dtm_std = np.nanstd(self.dtm[rcindices])
            dsm_max = np.nanmax(self.dsm[rcindices])

            if np.isnan(dtm_mean) or np.isnan(dsm_max):
                self.trees.tt_corrected.iloc[tidx] = -1
                continue

            # check if tree top too far down-slope compared to crown_mean
            if self.dtm[row, col] <= (dtm_mean - dtm_std) or check_all:

                # find highest DSM location in crown
                midx = np.where(self.dsm[rcindices] == dsm_max)[0][0]
                dsmhigh = np.array((rcindices[0][midx] + 0.5,
                                    rcindices[1][midx] + 0.5))

                # calculate map distances
                distances = cdist(np.stack(rcindices, axis=1),
                                  comass[tidx][np.newaxis])
                dist_dh_com = cdist(dsmhigh[np.newaxis],
                                    comass[tidx][np.newaxis])

                # assign high point position from DSM if new location is not
                # too far from centre of mass of the crown, in the latter case
                # place the tree top at the centre of mass
                corr_n += 1

                if dist_dh_com <= (1. * np.nanmean(distances)):
                    cor_col, cor_row = dsmhigh[1], dsmhigh[0]
                    corr_dsm += 1
                    self.trees.tt_corrected.iloc[tidx] = 1
                else:
                    cor_col, cor_row = comass[tidx][1], comass[tidx][0]
                    corr_com += 1
                    self.trees.tt_corrected.iloc[tidx] = 2

                # Set new tree top height
                self.trees.top_cor.iloc[tidx] = \
                    Point(*self._to_lonlat(cor_col, cor_row, self.resolution))

            else:
                self.trees.tt_corrected.iloc[tidx] = 0

        print(f'Tree tops corrected: {corr_n}')
        if len(self.trees) > 0:
            print(f'Tree tops corrected: {100 * corr_n / len(self.trees)}%')
            print(f'DSM correction: {corr_dsm}')
            print(f'COM correction: {corr_com}')
        return corr_dsm, corr_com

    def screen_small_trees(self, hmin=20., loc='top'):
        """ Remove small trees from index based on minimum threshold.
        Tree dataframe and crowns raster is updated.

        Parameters
        ----------
        hmin :    float
                  minimum height of tree top
        loc :     str, optional
                  tree seed position: `top` or `top_cor`
        """
        cond = self.trees[f'{loc}_height'] >= hmin
        self.trees = self.trees[cond]
        self.trees = self.trees.reset_index(drop=True)
        if isinstance(self.crowns, np.ndarray):
            self._screen_crowns(cond)

        self._check_empty()

    def crowns_to_polys_raster(self):
        ''' Converts tree crown raster to individual polygons and stores them
        in the tree dataframe
        '''
        polys = []
        for feature in rioshapes(self.crowns, mask=self.crowns.astype(bool)):
            # Convert pixel coordinates to lon/lat

            edges = feature[0]['coordinates'][0].copy()
            for i in range(len(edges)):
                edges[i] = self._to_lonlat(*edges[i], self.resolution)

            # poly_smooth = self.smooth_poly(Polygon(edges), s=None, k=9)
            polys.append(Polygon(edges))
        self.trees.crown_poly_raster = polys

    def crowns_to_polys_smooth(self, store_las=True, thin_perc=None,
                               first_return=True):
        """ Smooth crown polygons using Dalponte & Coomes (2016) approach:
        Builds a convex hull around first return points (which lie within the
        rasterized crowns).
        Optionally, the trees in the LiDAR point cloud are classified based on
        the generated convex hull.

        Parameters
        ----------
        store_las :    bool
                       set to True if LiDAR point clouds shopuld be classified
                       and stored externally
        thin_perc :    None or int
                       percentage amount by how much the point cloud should be
                       thinned out randomly
        first_return : bool
                       use first return points to create convex hull (all
                       points otherwise)
        """

        if thin_perc:
            thin_size = floor(len(self.las) * (1 - thin_perc))
            lidar_geodf = self.las.sample(n=thin_size)
        else:
            lidar_geodf = self.las

        print('Converting LAS point cloud to shapely points')
        geometry = [Point(xy) for xy in zip(lidar_geodf.x, lidar_geodf.y)]
        lidar_geodf = gpd.GeoDataFrame(lidar_geodf, crs=f'epsg:{self.epsg}',
                                       geometry=geometry)

        print('Converting raster crowns to shapely polygons')
        polys = []
        for feature in rioshapes(self.crowns, mask=self.crowns.astype(bool)):
            edges = np.array(list(zip(*feature[0]['coordinates'][0])))
            edges = np.array(self._to_lonlat(edges[0], edges[1],
                                             self.resolution)).T
            polys.append(Polygon(edges))
        crown_geodf = gpd.GeoDataFrame(
            pd.DataFrame(np.arange(len(self.trees))),
            crs=f'epsg:{self.epsg}', geometry=polys
        )

        print('Attach LiDAR points to corresponding crowns')
        lidar_in_crowns = gpd.sjoin(lidar_geodf, crown_geodf,
                                    predicate='within', how="inner")

        lidar_tree_class = np.zeros(lidar_in_crowns['index_right'].size)
        lidar_tree_mask = np.zeros(lidar_in_crowns['index_right'].size,
                                   dtype=bool)

        print('Create convex hull around first return points')
        polys = []
        for tidx in range(len(self.trees)):
            bool_indices = lidar_in_crowns['index_right'] == tidx
            lidar_tree_class[bool_indices] = tidx
            points = lidar_in_crowns[bool_indices]
            # check that not all values are the same
            if len(points.z) > 1 and not np.allclose(points.z,
                                                     points.iloc[0].z):
                z_values = np.array(points.z)
                points = points[points.z >= threshold_otsu(z_values)]
                if first_return:
                    points = points[points.return_num == 1]  # first returns
            hull = points.unary_union.convex_hull
            polys.append(hull)
            lidar_tree_mask[bool_indices] = \
                lidar_in_crowns[bool_indices].within(hull)
        self.trees.crown_poly_smooth = polys

        if store_las:
            print('Classifying point cloud')
            lidar_in_crowns = lidar_in_crowns[lidar_tree_mask]
            lidar_tree_class = lidar_tree_class[lidar_tree_mask]
            header = laspy.LasHeader(point_format=3, version="1.4")
            self.outpath.mkdir(parents=True, exist_ok=True)
            with laspy.open(self.outpath / "trees.las", mode="w", header=header) as outfile:
                # Compute the minimum values for offsets
                xmin = np.floor(np.min(lidar_in_crowns.x))
                ymin = np.floor(np.min(lidar_in_crowns.y))
                zmin = np.floor(np.min(lidar_in_crowns.z))
                # Set the header offsets and scales
                outfile.header.offset = [xmin, ymin, zmin]
                outfile.header.scale = [0.001, 0.001, 0.001]
                # Create the point records
                outfile.x = lidar_in_crowns.x
                outfile.y = lidar_in_crowns.y
                outfile.z = lidar_in_crowns.z
                outfile.intensity = lidar_tree_class

        self.lidar_in_crowns = lidar_in_crowns

    def quality_control(self, all_good=False):
        """ Remove trees from tree dataframe with missing DTM/DSM data &
        crowns that are not polygons

        Parameters
        ----------
        all_good :    bool
                      set to True if all trees should pass the quality check
        """
        if all_good:
            self.trees.tt_corrected = np.zeros(len(self.trees), dtype=int)
        else:
            cond = (
                    (self.trees.tt_corrected >= 0) &
                    self.trees.crown_poly_raster.apply(
                        lambda x: isinstance(x, Polygon))
            )
            self.trees = self.trees[cond]

        self._check_empty()

    def export_tree_locations(self, loc='top'):
        """ Convert tree top raster indices to georeferenced 3D point shapefile

        Parameters
        ----------
        loc :     str, optional
                  tree seed position: `top` or `top_cor`
        """
        outfile = self.outpath / f'tree_location_{loc}.shp'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        if outfile.exists():
            outfile.unlink()

        schema = {
            'geometry': '3D Point',
            'properties': {'DN': 'int', 'TH': 'float', 'COR': 'int'}
        }
        with fiona.collection(
                str(outfile), 'w', 'ESRI Shapefile', schema, crs=self.srs
        ) as output:
            for tidx in range(len(self.trees)):
                feat = {}
                tree = self.trees.iloc[tidx]
                feat['geometry'] = mapping(
                    Point(tree[loc].x, tree[loc].y, tree[f'{loc}_elevation'])
                )
                feat['properties'] = {'DN': tidx,
                                      'TH': float(tree[f'{loc}_height']),
                                      'COR': int(tree.tt_corrected)}
                output.write(feat)

    def export_tree_crowns(self, crowntype='crown_poly_smooth'):
        """ Convert tree crown raster to georeferenced polygon shapefile

        Parameters
        ----------
        crowntype :   str, optional
                      choose whether the raster of smoothed version should be
                      exported: `crown_poly_smooth` or `crown_poly_raster`
        """
        outfile = self.outpath / f'tree_{crowntype}.shp'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        if outfile.exists():
            outfile.unlink()

        schema = {
            'geometry': 'Polygon',
            'properties': {'DN': 'int', 'TTH': 'float', 'TCH': 'float'}
        }
        with fiona.collection(
                str(outfile), 'w', 'ESRI Shapefile',
                schema, crs=self.srs
        ) as output:
            for tidx in range(len(self.trees)):
                feat = {}
                tree = self.trees.iloc[tidx]

                # Check and fix the geometry if needed
                geometry = tree[crowntype]
                if isinstance(geometry, GeometryCollection):
                    if len(geometry.geoms) > 0:
                        # Extract the first Polygon geometry from the GeometryCollection
                        # print(geometry.geoms)
                        # geometry = geometry.geoms[0]  # Or handle as needed (e.g., take all polygons)
                        for geom in geometry.geoms:
                            feat['geometry'] = mapping(geom)
                            feat['properties'] = {
                                'DN': tidx,
                                'TTH': float(tree.top_height),
                                'TCH': float(tree.top_cor_height)
                            }

                            # Write the feature to the shapefile
                            output.write(feat)
                        continue
                    else:
                        # Handle case where the GeometryCollection is empty
                        continue  # Skip this tree or add handling logic as necessary.
                if isinstance(geometry, LineString):
                    # Convert LineString to Polygon using a small buffer
                    geometry = geometry.buffer(0.01)  # Adjust the buffer size as needed
                if isinstance(geometry, Point):
                    geometry = geometry.buffer(0.5)

                # Prepare feature with geometry and properties
                feat['geometry'] = mapping(geometry)
                feat['properties'] = {
                    'DN': tidx,
                    'TTH': float(tree.top_height),
                    'TCH': float(tree.top_cor_height)
                }

                # Write the feature to the shapefile
                output.write(feat)

    def export_tree_indexes(self, ndvi_file, savi_file, evi_file, reci_file, gndvi_file):

        ndvi_raster = rasterio.open(ndvi_file)
        savi_raster = rasterio.open(savi_file)
        evi_raster = rasterio.open(evi_file)
        reci_raster = rasterio.open(reci_file)
        gndvi_raster = rasterio.open(gndvi_file)

        # indexes per tree
        loc = 'top'
        crowntype = 'crown_poly_raster'

        indexes = []
        for tidx in range(len(self.trees)):
            tree = self.trees.iloc[tidx]

            geometry = tree[crowntype]

            if isinstance(geometry, GeometryCollection):
                if len(geometry.geoms) > 0:
                    # Extract the first Polygon geometry from the GeometryCollection
                    # print(geometry.geoms)
                    # geometry = geometry.geoms[0]  # Or handle as needed (e.g., take all polygons)
                    geometry_ = []
                    for geom in geometry.geoms:
                        geometry_.append(mapping(geom))
                    geometry = geometry_
                    continue
                else:
                    # Handle case where the GeometryCollection is empty
                    continue  # Skip this tree or add handling logic as necessary.
            elif isinstance(geometry, LineString):
                # Convert LineString to Polygon using a small buffer
                geometry = geometry.buffer(0.01)  # Adjust the buffer size as needed
                geometry = [mapping(geometry)]
            elif isinstance(geometry, Point):
                geometry = geometry.buffer(0.5)
                geometry = [mapping(geometry)]
            else:
                geometry = [geometry]

            sum_ndvi = 0
            sum_savi = 0
            sum_evi = 0
            sum_reci = 0
            sum_gndvi = 0
            perimeter = 0
            area = 0
            for geometry_i in geometry:
                # Mask the raster with the polygon
                masked_raster_ndvi, transform = mask(ndvi_raster, [geometry_i], crop=True)
                masked_raster_savi, transform = mask(savi_raster, [geometry_i], crop=True)
                masked_raster_evi, transform = mask(evi_raster, [geometry_i], crop=True)
                masked_raster_reci, transform = mask(reci_raster, [geometry_i], crop=True)
                masked_raster_gndvi, transform = mask(gndvi_raster, [geometry_i], crop=True)

                pixel_values = masked_raster_ndvi[0].flatten()
                pixel_values = pixel_values[pixel_values != ndvi_raster.nodata]
                sum_ndvi += np.mean(pixel_values)

                pixel_values = masked_raster_savi[0].flatten()
                pixel_values = pixel_values[pixel_values != savi_raster.nodata]
                sum_savi += np.mean(pixel_values)

                pixel_values = masked_raster_evi[0].flatten()
                pixel_values = pixel_values[pixel_values != evi_raster.nodata]
                sum_evi += np.mean(pixel_values)

                pixel_values = masked_raster_reci[0].flatten()
                pixel_values = pixel_values[pixel_values != reci_raster.nodata]
                sum_reci += np.mean(pixel_values)

                pixel_values = masked_raster_gndvi[0].flatten()
                pixel_values = pixel_values[pixel_values != gndvi_raster.nodata]
                sum_gndvi += np.mean(pixel_values)
                perimeter = geometry_i.length
                area = geometry_i.area

            avg_ndvi = sum_ndvi / len(geometry)
            avg_savi = sum_savi / len(geometry)
            avg_evi = sum_evi / len(geometry)
            avg_reci = sum_reci / len(geometry)
            avg_gndvi = sum_gndvi / len(geometry)

            dic_tidx = {"X": tree[loc].x, "Y": tree[loc].y, "Elevation": tree[f'{loc}_elevation'].item(), "Height": tree[f'{loc}_height'].item(),
                        "perimetro": str(perimeter), "area": str(area),"ndvi": avg_ndvi.item(), "savi": avg_savi.item(), "evi": avg_evi.item(), "reci": avg_reci.item(),
                        "gndvi": avg_gndvi.item()}
            indexes.append(dic_tidx)

        return indexes

    def export_raster(self, raster, fname, title, res=None):
        """ Write array to raster file with gdal

        Parameters
        ----------
        raster :    ndarray
                    raster to be exported
        fname :     str
                    file name
        title :     str
                    gdal title of the file
        res :       int/float, optional
                    resolution of the raster in m, if not provided the same as
                    the input CHM
        """
        res = res if res else self.resolution

        fname.parent.mkdir(parents=True, exist_ok=True)
        try:
            driver = gdal.GetDriverByName('GTIFF')
            y_pixels, x_pixels = raster.shape
            gdal_file = driver.Create(
                f'{fname}', x_pixels, y_pixels, 1, gdal.GDT_Float32
            )
            gdal_file.SetGeoTransform(
                (self.ul_lon, res, 0., self.ul_lat, 0., -res)
            )
            dataset_srs = gdal.osr.SpatialReference()
            dataset_srs.ImportFromEPSG(self.epsg)
            gdal_file.SetProjection(dataset_srs.ExportToWkt())
            band = gdal_file.GetRasterBand(1)
            band.SetDescription(title)
            band.SetNoDataValue(0.)
            band.WriteArray(raster)
            gdal_file.FlushCache()
            gdal_file = None
        except:
            ul_lon = self.ul_lon
            ul_lat = self.ul_lat
            title = 'trees'  # Band description
            epsg_code = self.epsg  # EPSG code for the coordinate system
            # Define the transform (upper-left corner and pixel size)
            transform = from_origin(ul_lon, ul_lat, res,
                                    res)  # (upper left x, upper left y, pixel size x, pixel size y)
            # Open a new GeoTIFF file for writing
            with rasterio.open(
                    f'{fname}', 'w',
                    driver='GTiff',
                    count=1,
                    dtype='float32',
                    width=raster.shape[1],
                    height=raster.shape[0],
                    crs=CRS.from_epsg(epsg_code),
                    transform=transform
            ) as dst:
                # Write the array to the file
                dst.write(raster, 1)  # 1 refers to the first band

                # Set the NoData value
                dst.nodata = 0.0

                # Set the band description (optional)
                dst.descriptions = (title,)
