"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import logging
import os
import sys
import warnings
from ctypes import cdll
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import geoCosiCorr3D.geoCore.constants as const
import geoCosiCorr3D.geoImageCorrelation.misc as misc
import numpy as np
from geoCosiCorr3D.geoCore.base.base_correlation import (BaseCorrelation,
                                                         BaseCorrelationEngine,
                                                         BaseFreqCorr,
                                                         BaseSpatialCorr)
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo
from skimage.feature import match_template #zhuang
import pywt #zhuang
import dtcwt

FREQ_CORR_LIB = const.CORRELATION.FREQ_CORR_LIB
STAT_CORR_LIB = const.CORRELATION.STAT_CORR_LIB


# TODO:
#  1- add flag to perform pixel-based correlation
#  2- Base class for overlap based on projection and based on pixel, for the pixel based we need support x_off and y_off
#  3- Support: Optical flow correlation, MicMac, ASP, Sckit-image, OpenCV corr, ....
#  4- Data sets: geometry artifacts (PS, HiRISE,WV, GF), glacier , cloud detection, earthquake, landslide, dune

# define Python user-defined exceptions
class InvalidCorrLib(Exception):
    pass


class RawFreqCorr(BaseFreqCorr):
    def __init__(self,
                 window_size: List[int] = None,
                 step: List[int] = None,
                 mask_th: float = None,
                 resampling: bool = False,
                 nb_iter: int = 4,
                 grid: bool = True):

        if window_size is None:
            self.window_size = 4 * [64]
        else:
            self.window_size = window_size

        if step is None:
            self.step = 2 * [8]
        else:
            self.step = step
        if mask_th is None:
            self.mask_th = 0.9
        else:
            self.mask_th = mask_th
        self.resampling = resampling
        self.nb_iter = nb_iter
        self.grid = grid
        return

    @staticmethod
    def ingest_freq_corr_params(params: Dict) -> List[Any]:
        """

        Args:
            params:

        Returns:

        """
        window_size = params.get("window_size", 4 * [64])
        step = params.get("step", 2 * [8])
        mask_th = params.get("mask_th", 0.9)
        nb_iters = params.get("nb_iters", 4)
        grid = params.get("grid", True)
        return [window_size, step, mask_th, nb_iters, grid]

    @staticmethod
    def set_margins(resampling: bool, window_size: List[int]) -> List[int]:
        """

        Args:
            resampling:
            window_size:

        Returns:

        """
        if ~resampling:
            margins = [int(window_size[0] / 2), int(window_size[1] / 2)]
            logging.info("corr margins: {}".format(margins))
            return margins
        else:
            logging.warning("Compute margin based on resampling Kernel ! ")
            raise NotImplementedError

    # TODO: change to static method or adapt to class method
    @classmethod
    def run_correlator(cls, base_array: np.ndarray,
                       target_array: np.ndarray,
                       window_size: List[int],
                       step: List[int],
                       iterations: int,
                       mask_th: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            lib_cfreq_corr = cdll.LoadLibrary(FREQ_CORR_LIB)
        except:
            raise InvalidCorrLib
        lib_cfreq_corr.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.int32)]

        input_shape = np.array(base_array.shape, dtype=np.int32)
        window_sizes = np.array(window_size, dtype=np.int32)
        step_sizes = np.array(step, dtype=np.int32)
        base_array_ = np.array(base_array.flatten(), dtype=np.float32)
        target_array_ = np.array(target_array.flatten(), dtype=np.float32)
        ew_array_ = np.zeros((input_shape[0] * input_shape[1], 1), dtype=np.float32)
        ns_array_ = np.zeros((input_shape[0] * input_shape[1], 1), dtype=np.float32)
        snr_array_ = np.zeros((input_shape[0] * input_shape[1], 1), dtype=np.float32)
        iteration = np.array([iterations], dtype=np.int32)
        mask_threshold = np.array([mask_th], dtype=np.float32)
        output_shape = np.array([0, 0], dtype=np.int32)
        lib_cfreq_corr.InputData(input_shape,
                                 window_sizes,
                                 step_sizes,
                                 base_array_,
                                 target_array_,
                                 ew_array_,
                                 ns_array_,
                                 snr_array_,
                                 iteration,
                                 mask_threshold,
                                 output_shape)

        ew_array_fl = ew_array_[0:output_shape[0] * output_shape[1]]
        ns_array_fl = ns_array_[0:output_shape[0] * output_shape[1]]
        snr_array_fl = snr_array_[0:output_shape[0] * output_shape[1]]
        ew_array = np.asarray(ew_array_fl).reshape((output_shape[0], output_shape[1]))
        ns_array = np.asarray(ns_array_fl).reshape((output_shape[0], output_shape[1]))
        snr_array = np.asarray(snr_array_fl).reshape((output_shape[0], output_shape[1]))

        return ew_array, ns_array, snr_array


class RawSpatialCorr(BaseSpatialCorr):

    def __init__(self,
                 window_size: List[int] = None,
                 steps: List[int] = None,
                 search_range: List[int] = None,
                 grid: bool = False):
        """

        Args:
            window_size:
            steps:
            search_range:
            grid:
        """
        if window_size is None:
            self.window_size = 2 * [32]
        else:
            self.window_size = window_size
        if steps is None:
            self.step = 2 * [16]
        else:
            self.step = steps
        if search_range is None:
            self.search_range = 2 * [10]
        else:
            self.search_range = search_range

        self.grid = grid

        return

    @staticmethod
    def get_output_dims(step_size: List[int],
                        input_shape: Tuple[int, int],
                        window_size: List[int],
                        range_size: List[int]) -> Tuple[int, int]:
        """

        Args:
            step_size:
            input_shape:
            window_size:
            range_size:

        Returns:

        """

        if (step_size[0] != 0):
            value = input_shape[1] - (window_size[0] + 2 * range_size[0])
            output_cols = int((np.floor(value / step_size[0] + 1.0)))
        else:
            output_cols = 1
        if (step_size[1] != 0):

            value = (input_shape[0] - (window_size[1] + 2 * range_size[1]))
            output_rows = int(np.floor(value / step_size[1] + 1.0))
        else:
            output_rows = 1
        return (output_rows, output_cols)

    @classmethod
    def run_correlator(cls, base_array: np.ndarray,
                       target_array: np.ndarray,
                       window_size: List[int],
                       step: List[int],
                       search_range: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            libCstatCorr = cdll.LoadLibrary(STAT_CORR_LIB)
        except:
            raise InvalidCorrLib

        libCstatCorr.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32)]

        inputShape = np.array(base_array.shape, dtype=np.int32)
        windowSizes = np.array(window_size, dtype=np.int32)
        stepSizes = np.array(step, dtype=np.int32)
        searchRanges = np.array(search_range, dtype=np.int32)
        baseArray = np.array(base_array.flatten(), dtype=np.float32)
        targetArray = np.array(target_array.flatten(), dtype=np.float32)

        outputRows, outputCols = cls.get_output_dims(step_size=step,
                                                     input_shape=base_array.shape,
                                                     window_size=window_size,
                                                     range_size=search_range)
        outputShape = np.array([outputRows, outputCols], dtype=np.int32)

        ewArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)
        nsArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)
        snrArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)

        libCstatCorr.InputData(inputShape, windowSizes, stepSizes, searchRanges, baseArray, targetArray, outputShape,
                               ewArray_fl, nsArray_fl, snrArray_fl)

        ew_array = np.asarray(ewArray_fl[:, 0]).reshape((outputRows, outputCols))
        ns_Array = np.asarray(nsArray_fl[:, 0]).reshape((outputRows, outputCols))
        snr_array = np.asarray(snrArray_fl[:, 0]).reshape((outputRows, outputCols))

        return ew_array, ns_Array, snr_array

    @staticmethod
    def ingest_spatial_corr_params(params: Dict) -> List[Any]:
        window_size = params.get("window_size", [64, 64, 64, 64])
        step = params.get("step", 2 * [8])
        search_range = params.get("search_range", 2 * [10])
        grid = params.get("grid", True)
        return [window_size, step, search_range, grid]


class RawCorrelationEngine(BaseCorrelationEngine):
    def __init__(self,
                 correlator_name: str = None,
                 params=None,
                 debug=False):

        self.correlator_name = correlator_name

        if self.correlator_name is None:
            self.correlator_name = const.CORRELATION.FREQUENCY_CORRELATOR
        self.corr_params = params
        self.debug = debug
        self.corr_bands: List[str] = ["East/West", "North/South", "SNR"]

        self._get_corr_params()

    def _get_corr_params(self):
        if self.correlator_name == const.CORRELATION.FREQUENCY_CORRELATOR:
            if self.corr_params is None:
                self.corr_params = self.get_freq_params()
            else:
                self.corr_params = self.get_freq_params(window_size=self._ingest_params()[0],
                                                        step=self._ingest_params()[1],
                                                        mask_th=self._ingest_params()[2],
                                                        nb_iters=self._ingest_params()[3],
                                                        grid=self._ingest_params()[4])
        if self.correlator_name == const.CORRELATION.SPATIAL_CORRELATOR:
            if self.corr_params is None:
                self.corr_params = self.get_spatial_params()
            else:
                self.corr_params = self.get_spatial_params(window_size=self._ingest_params()[0],
                                                           step=self._ingest_params()[1],
                                                           search_range=self._ingest_params()[2],
                                                           grid=self._ingest_params()[3])
        
        if self.correlator_name == const.CORRELATION.NCC_CORRELATOR:
            if self.corr_params is None:
                # same defaults as spatial
                self.corr_params = self.get_ncc_params()
            else:
                self.corr_params = self.get_ncc_params(window_size=self._ingest_params()[0],
                                                       step=self._ingest_params()[1],
                                                       search_range=self._ingest_params()[2],
                                                       grid=self._ingest_params()[3])
                                                       
        if self.correlator_name == const.CORRELATION.WAVELET_NCC_CORRELATOR:
            if self.corr_params is None:
                self.corr_params = self.get_wavelet_ncc_params()
            else:
                # reuse spatial ingester to pull window/step/search/grid out of the dict
                w, s, r, g = RawSpatialCorr.ingest_spatial_corr_params(self.corr_params)
                self.corr_params = self.get_wavelet_ncc_params(
                    window_size=w, step=s, search_range=r, grid=g,
                    wavelet=self.corr_params.get('wavelet', 'db2'),
                    levels=int(self.corr_params.get('levels', 3)),
                    min_tpl_var=float(self.corr_params.get('min_tpl_var', 1e-4)),
                    min_peak=float(self.corr_params.get('min_peak', 0.5)),
                )
        if self.correlator_name == const.CORRELATION.DTCWT_NCC_CORRELATOR:
            if self.corr_params is None:
                self.corr_params = self.get_dtcwt_ncc_params()
            else:
                w, s, r, g = RawSpatialCorr.ingest_spatial_corr_params(self.corr_params)
                self.corr_params = self.get_dtcwt_ncc_params(
                    window_size=w, step=s, search_range=r, grid=g,
                    levels=int(self.corr_params.get('levels', 4)),
                    min_tpl_var=float(self.corr_params.get('min_tpl_var', 1e-4)),
                    min_peak=float(self.corr_params.get('min_peak', 0.5)),
                )

        if self.debug:
            logging.info(self.__dict__)
        return

    @staticmethod
    def get_spatial_params(window_size: List[int] = None, step: List[int] = None,
                           search_range: List[int] = None, grid: bool = False) -> RawSpatialCorr:

        return RawSpatialCorr(window_size, step, search_range, grid)

    @staticmethod
    def get_freq_params(window_size: List[int] = None, step: List[int] = None, mask_th: float = None,
                        resampling: bool = False, nb_iters: int = 4, grid: bool = True) -> RawFreqCorr:

        return RawFreqCorr(window_size, step, mask_th, resampling, nb_iters, grid)

    @staticmethod
    def get_ncc_params(window_size: list = None, step: list = None,
                       search_range: list = None, grid: bool = True) -> "RawNCCCorr":
        return RawNCCCorr(window_size, step, search_range, grid)

    @staticmethod
    def get_wavelet_ncc_params(window_size: list = None, step: list = None,
                            search_range: list = None, grid: bool = True,
                            wavelet: str = 'db2', levels: int = 3,
                            min_tpl_var: float = 1e-4, min_peak: float = 0.5) -> "WaveletNCCCorr":
        # we pass config when calling run_correlator, so here we can just instantiate
        return WaveletNCCCorr(window_size, step, search_range, grid,
                            wavelet=wavelet, levels=levels,
                            min_tpl_var=min_tpl_var, min_peak=min_peak)

    @staticmethod
    def get_dtcwt_ncc_params(window_size: list = None, step: list = None,
                            search_range: list = None, grid: bool = True,
                            levels: int = 4, min_tpl_var: float = 1e-4,
                            min_peak: float = 0.5) -> "DTCWTNCCCorr":
        return DTCWTNCCCorr(window_size, step, search_range, grid,
                        levels=levels, min_tpl_var=min_tpl_var,
                        min_peak=min_peak)

    def _ingest_params(self):
        if self.correlator_name == const.CORRELATION.FREQUENCY_CORRELATOR:
            return RawFreqCorr.ingest_freq_corr_params(params=self.corr_params)
        if self.correlator_name == const.CORRELATION.SPATIAL_CORRELATOR:
            return RawSpatialCorr.ingest_spatial_corr_params(params=self.corr_params)

        if self.correlator_name == const.CORRELATION.WAVELET_NCC_CORRELATOR:
            return RawSpatialCorr.ingest_spatial_corr_params(params=self.corr_params)
        if self.correlator_name == const.CORRELATION.NCC_CORRELATOR:
            return RawSpatialCorr.ingest_spatial_corr_params(params=self.corr_params)

        if self.correlator_name == const.CORRELATION.DTCWT_NCC_CORRELATOR:
            return RawSpatialCorr.ingest_spatial_corr_params(params=self.corr_params)
            
    def correlate(self):
        pass


class CorrelationEngine(RawCorrelationEngine):
    def __init__(self, correlator_name: str = None,
                 params=None,
                 debug=False):
        super().__init__(correlator_name,
                         params,
                         debug)
        pass


class RawCorrelation(BaseCorrelation):

    def __init__(self,
                 base_image_path: str,
                 target_image_path: str,
                 corr_config: Optional[Dict] = None,
                 base_band: Optional[int] = 1,
                 target_band: Optional[int] = 1,
                 output_corr_path: Optional[str] = None,
                 tile_size_mb: Optional[int] = const.CORRELATION.TILE_SIZE_MB,
                 visualize: Optional[bool] = False,
                 debug: Optional[bool] = True,
                 pixel_based_correlation: Optional[bool] = None):

        self.snr_output = None
        self.ns_output = None
        self.ew_output = None
        self.win_area_x = None
        self.win_area_y = None
        self.base_img_path = base_image_path
        self.target_img_path = target_image_path
        self.corr_engine = CorrelationEngine(correlator_name=corr_config.get("correlator_name", None),
                                             params=corr_config.get("correlator_params", None))

        self.tile_size_mb = tile_size_mb
        self.visualize = visualize
        self.base_band_nb = base_band
        self.target_band_nb = target_band
        self.output_corr_path = output_corr_path
        self.debug = debug
        self.pixel_based_correlation = pixel_based_correlation

    def _ingest(self):

        self.x_res: float
        self.y_res: float
        self.win_area_x: int
        self.win_area_y: int
        self.base_dims_pix: List[float]
        self.target_dims_pix: List[float]
        self.tot_col: int
        self.tot_row: int

        if self.output_corr_path is None:
            self.output_corr_path = self.make_output_path(os.path.dirname(self.base_img_path), self.base_img_path,
                                                          self.target_img_path, self.corr_engine.correlator_name,
                                                          self.corr_engine.corr_params.window_size[0],
                                                          self.corr_engine.corr_params.step[0])
        else:
            if os.path.isdir(self.output_corr_path):
                self.output_corr_path = self.make_output_path(self.output_corr_path, self.base_img_path,
                                                              self.target_img_path, self.corr_engine.correlator_name,
                                                              self.corr_engine.corr_params.window_size[0],
                                                              self.corr_engine.corr_params.step[0])

        if self.debug:
            logging.info("Correlation engine:{} , params:{}".format(self.corr_engine.correlator_name,
                                                                    self.corr_engine.corr_params.__dict__))

        self.base_info = cRasterInfo(self.base_img_path)
        self.target_info = cRasterInfo(self.target_img_path)
        # TODo remove the -1 form the list and use rasterio or bbox_pix bbox_map instead
        self.base_dims_pix = [-1, 0, int(self.base_info.raster_width) - 1, 0,
                              int(self.base_info.raster_height) - 1]
        self.target_dims_pix = [-1, 0, int(self.target_info.raster_width) - 1, 0,
                                int(self.target_info.raster_height) - 1]

        self.base_original_dims: List[float] = self.base_dims_pix
        self.margins = self.set_margins()
        logging.info(f'{self.__class__.__name__}:correlation margins:{self.margins}')
        if self.pixel_based_correlation is None:
            self.pixel_based_correlation = False
        return

    def set_margins(self) -> List[int]:
        if self.corr_engine.correlator_name == const.CORRELATION.FREQUENCY_CORRELATOR:
            return RawFreqCorr.set_margins(self.corr_engine.corr_params.resampling,
                                           self.corr_engine.corr_params.window_size)
        if self.corr_engine.correlator_name in (
            const.CORRELATION.SPATIAL_CORRELATOR,
            const.CORRELATION.NCC_CORRELATOR,
            const.CORRELATION.WAVELET_NCC_CORRELATOR,   # zhuang
            const.CORRELATION.DTCWT_NCC_CORRELATOR,
        ):
            return self.corr_engine.corr_params.search_range

    def check_same_projection_system(self):
        # TODO: move this function misc
        ##Check that the images have identical projection reference system
        self.flags = {"validMaps": False, "groundSpaceCorr": False, "continue": True}
        if self.pixel_based_correlation:
            logging.info(' USER: PIXEL-BASED CORRELATION ')
            self.flagList = self.updateFlagList(self.flags)
            return
        if self.base_info.valid_map_info and self.target_info.valid_map_info:
            self.flags["validMaps"] = True
            if self.base_info.epsg_code != self.target_info.epsg_code:
                # TODO: Check based on the projection not only epsg_code
                # TODO: add the possibility to reproject to the same projection system + same resolution
                logging.warning(
                    "=== Input images have different map projection (!= EPSG code), Correlation will be pixel based! ===")
                # warnings.warn(
                #     "=== Input images have different map projection (!= EPSG code), Correlation will be pixel based! ===",
                #     stacklevel=2)
                self.flags["groundSpaceCorr"] = False
            else:
                self.flags["groundSpaceCorr"] = True
        else:
            logging.warning("=== Input images are not geo-referenced. Correlation will be pixel based!,  ===")
            warnings.warn(
                "=== Input images are not geo-referenced. Correlation will be pixel based!,  ===",
                stacklevel=2)
            self.flags["groundSpaceCorr"] = False

        self.flagList = self.updateFlagList(self.flags)

    def check_same_ground_resolution(self):
        ## Check if the images have the same ground resolution
        # (up to 1/1000 of the resolution to avoid precision error)
        if all(self.flagList):
            same_res = misc.check_same_gsd(base_img_info=self.base_info, target_img_info=self.target_info)
            if same_res == False:
                self.flags["continue"] = False
                # TODO: add error output -> raise error
                logging.error("=== Images have the same GSD:{} ===".format(self.base_info.pixel_width))
                logging.error("=== ERROR: Input data must have the same resolution to be correlated ===")
                sys.exit("=== ERROR: Input data must have the same resolution to be correlated ===")
            else:
                if self.debug:
                    logging.warning("=== Images have the same GSD:{} ===".format(self.base_info.pixel_width))
        self.flagList = self.updateFlagList(self.flags)

    def _check_aligned_grids(self):
        ## Check that the imqges are on geographically aligned grids (depends on origin and resolution)
        ## verify if the difference between image origin is less than of resolution/1000

        if all(self.flagList):
            if misc.check_aligned_grids(base_img_info=self.base_info, target_img_info=self.target_info) == False:
                self.flags["overlap"] = False
                # TODO raise ana error
                ## Add the possibility to align the inpu grids
                error_msg = "=== ERROR: --- Images cannot be overlapped due to their origin and resolution - " \
                            "Origins difference great than 1/1000 of the pixel resolution ==="
                logging.error(error_msg)
                sys.exit(error_msg)
            else:
                self.flags["overlap"] = True
        self.flagList = self.updateFlagList(self.flags)

    def set_corr_map_resolution(self):
        ## Depending on the validity of the map information of the images, the pixel resolution is setup
        # it will be the GSD if map info is valid , otherwise it will be 1 and the correlation will be pixel based
        if all(self.flagList):
            # the correlation will be map based not pixel based
            self.y_res = np.abs(self.base_info.pixel_height)
            self.x_res = np.abs(self.base_info.pixel_width)
        else:
            # Correlation will be pixel based
            self.x_res = 1.0
            self.y_res = 1.0

    @staticmethod
    def _set_win_area(window_sizes: List[int], margins: List[int]):
        """

        Args:
            window_sizes:
            margins:

        Returns:

        """
        win_area_x = int(window_sizes[0] + 2 * margins[0])
        win_area_y = int(window_sizes[1] + 2 * margins[1])

        return win_area_x, win_area_y

    @staticmethod
    def _blank_array_func(nbVals: int, nbBands: Optional[int] = 3) -> Any:
        blank_arr = np.zeros((nbVals * nbBands))
        for i in range(nbVals):
            blank_arr[3 * i] = np.nan
            blank_arr[3 * i + 1] = np.nan
            blank_arr[3 * i + 2] = 0

        return blank_arr

    @staticmethod
    def updateFlagList(flagDic):
        return list(flagDic.values())

    def crop_to_same_size(self):
        # TODO: move this function to misc
        """
        Cropping the images to the same size:
        Two condition exist:
            1- if the map information are valid and identical: we define the overlapping area based on geo-referencing
            2- if map information invalid or different: we define the overlapping area based we define overlapping area
                based on images size (pixel wise)
                """
        if all(self.flagList):
            # If map information are valid and identical, define the overlapping are based on geo-referencing.
            # Backup of the original master subset dimensions in case of a non gridded correlation
            # IF NOT grid THEN img1OriginalDims = base_dims_pix
            offset = ((self.target_info.x_map_origin + self.target_dims_pix[1] * self.target_info.pixel_width) - (
                    self.base_info.x_map_origin + self.base_dims_pix[
                1] * self.base_info.pixel_width)) / self.base_info.pixel_width
            if offset > 0:
                self.base_dims_pix[1] = int(self.base_dims_pix[1] + round(offset))
            else:
                self.target_dims_pix[1] = int(self.target_dims_pix[1] - round(offset))

            offset = ((self.target_info.x_map_origin + self.target_dims_pix[2] * self.target_info.pixel_width) - (
                    self.base_info.x_map_origin + self.base_dims_pix[
                2] * self.base_info.pixel_width)) / self.base_info.pixel_width

            if offset < 0:
                self.base_dims_pix[2] = int(self.base_dims_pix[2] + round(offset))
            else:
                self.target_dims_pix[2] = int(self.target_dims_pix[2] - round(offset))

            offset = ((self.target_info.y_map_origin - self.target_dims_pix[3] * np.abs(
                self.target_info.pixel_height)) - (
                              self.base_info.y_map_origin - self.base_dims_pix[3] * np.abs(
                          self.base_info.pixel_height))) / np.abs(
                self.base_info.pixel_height)
            if offset < 0:
                self.base_dims_pix[3] = int(self.base_dims_pix[3] - round(offset))
            else:
                self.target_dims_pix[3] = int(self.target_dims_pix[3] + round(offset))

            offset = ((self.target_info.y_map_origin - self.target_dims_pix[4] * np.abs(
                self.target_info.pixel_height)) - (
                              self.base_info.y_map_origin - self.base_dims_pix[4] * np.abs(
                          self.base_info.pixel_height))) / np.abs(
                self.base_info.pixel_height)
            if offset > 0:
                self.base_dims_pix[4] = int(self.base_dims_pix[4] - round(offset))
            else:
                self.target_dims_pix[4] = int(self.target_dims_pix[4] + round(offset))

            if self.base_dims_pix[0] >= self.base_dims_pix[2] or self.base_dims_pix[3] >= self.base_dims_pix[4]:
                logging.error("=== ERROR: Images do not have a geographic overlap ===")
                sys.exit(
                    "=== ERROR: Images do not have a geographic overlap ===")

        else:
            # If map information invalid or different, define overlapping area based on images size (pixel-wise)

            if (self.base_dims_pix[2] - self.base_dims_pix[1]) > (self.target_dims_pix[2] - self.target_dims_pix[1]):
                self.base_dims_pix[2] = self.base_dims_pix[1] + (self.target_dims_pix[2] - self.target_dims_pix[1])
            else:
                self.target_dims_pix[2] = self.target_dims_pix[1] + (self.base_dims_pix[2] - self.base_dims_pix[1])
            if (self.base_dims_pix[4] - self.base_dims_pix[3]) > (self.target_dims_pix[4] - self.target_dims_pix[3]):
                self.base_dims_pix[4] = self.base_dims_pix[3] + (self.target_dims_pix[4] - self.target_dims_pix[3])
            else:
                self.target_dims_pix[4] = self.target_dims_pix[3] + (self.base_dims_pix[4] - self.base_dims_pix[3])

    def adjusting_cropped_images_according_2_grid_nongrid(self):
        # TODO: refactoring + split into 2 functions or class
        """
          Adjusting cropped images according to a gridded/non-gridded output
          Two cases:
              1- If the user selected the gridded option
              2- If the user selected non-gridded option
              """
        # If the user selected the gridded option
        if self.corr_engine.corr_params.grid:
            if all(self.flagList):
                if misc.decimal_mod(value=self.base_info.x_map_origin, param=self.base_info.pixel_width) != 0 or \
                        misc.decimal_mod(value=self.base_info.y_map_origin,
                                         param=np.abs(self.base_info.pixel_height)) != 0:
                    logging.error(
                        "=== ERROR: Images coordinates origins must be a multiple of the resolution for a gridded output' ===")
                    sys.exit(
                        "=== ERROR: Images coordinates origins must be a multiple of the resolution for a gridded output' ===")
                ## Chek if the geo-coordinate of the first correlated pixel is multiple integer of the resolution
                ## If not adjust the image boundaries
                geoOffsetX = (self.base_info.x_map_origin + (
                        self.base_dims_pix[1] + self.margins[0] + self.corr_engine.corr_params.window_size[
                    0] / 2) * self.base_info.pixel_width) % \
                             (self.corr_engine.corr_params.step[0] * self.base_info.pixel_width)

                # print(self.baseDims[1], self.corr.margins[0], self.corr.windowSizes[0] / 2, self.base_info.pixelWidth)

                if np.round(geoOffsetX / self.base_info.pixel_width) != 0:
                    self.base_dims_pix[1] = int(
                        self.base_dims_pix[1] + self.corr_engine.corr_params.step[0] - np.round(
                            geoOffsetX / self.base_info.pixel_width))
                    self.target_dims_pix[1] = int(
                        self.target_dims_pix[1] + self.corr_engine.corr_params.step[0] - np.round(
                            geoOffsetX / self.base_info.pixel_width))

                geoOffsetY = (self.base_info.y_map_origin - (
                        self.base_dims_pix[3] + self.margins[1] + self.corr_engine.corr_params.window_size[
                    1] / 2) * np.abs(
                    self.base_info.pixel_height)) % \
                             (self.corr_engine.corr_params.step[1] * np.abs(self.base_info.pixel_height))

                if np.round(geoOffsetY / np.abs(self.base_info.pixel_height)) != 0:
                    self.base_dims_pix[3] = int(
                        self.base_dims_pix[3] + np.round(geoOffsetY / np.abs(self.base_info.pixel_width)))
                    self.target_dims_pix[3] = int(
                        self.target_dims_pix[3] + np.round(geoOffsetY / np.abs(self.base_info.pixel_width)))

            ## Define the number of column and rows of the ouput correlation
            self.tot_col = int(np.floor(
                (self.base_dims_pix[2] - self.base_dims_pix[1] + 1 - self.corr_engine.corr_params.window_size[0] - 2
                 * self.margins[0]) / self.corr_engine.corr_params.step[0]) + 1)
            self.tot_row = int(np.floor(
                (self.base_dims_pix[4] - self.base_dims_pix[3] + 1 - self.corr_engine.corr_params.window_size[1] - 2 *
                 self.margins[
                     1]) /
                self.corr_engine.corr_params.step[1]) + 1)
            if self.debug:
                logging.info("tCols:{}, tRows:{}".format(self.tot_col, self.tot_row))


        else:
            # The non-gridded correlation will generate a correlation map whose first pixel corresponds
            # to the first master pixel

            # Define the total number of pixel of the correlation map in col and row
            self.tot_col = int(
                np.floor((self.base_original_dims[2] - self.base_original_dims[1]) / self.corr_engine.corr_params.step[
                    0]) + 1)
            self.tot_row = int(
                np.floor((self.base_original_dims[4] - self.base_original_dims[3]) / self.corr_engine.corr_params.step[
                    1]) + 1)
            if self.debug:
                logging.info("tCols:{}, tRows:{}".format(self.tot_col, self.tot_row))

            # Compute the "blank" border in col and row. This blank border corresponds to the area of the
            # correlation map where no correlation values could be computed due to the patch characteristic of the
            # correlator
            self.border_col_left: int = int(np.ceil(
                (self.base_dims_pix[1] - self.base_original_dims[1] + self.corr_engine.corr_params.window_size[0] / 2 +
                 self.margins[
                     0]) / float(self.corr_engine.corr_params.step[0])))
            self.border_row_top: int = int(np.ceil(
                (self.base_dims_pix[3] - self.base_original_dims[3] + self.corr_engine.corr_params.window_size[1] / 2 +
                 self.margins[
                     1]) / float(self.corr_engine.corr_params.step[1])))
            if self.debug:
                logging.info("borderColLeft:{}, borderRowTop:{}".format(self.border_col_left, self.border_row_top))
            # From the borders in col and row, compute the necessary cropping of the master and slave in row and col,
            # so the first patch retrived from the tile correponds to a step-wise position of the correlation grid origin
            offsetX = self.border_col_left * self.corr_engine.corr_params.step[0] - (
                    self.corr_engine.corr_params.window_size[0] / 2 + self.margins[0]) - (
                              self.base_dims_pix[1] - self.base_original_dims[1])
            offsetY = self.border_row_top * self.corr_engine.corr_params.step[1] - (
                    self.corr_engine.corr_params.window_size[1] / 2 + self.margins[1]) - (
                              self.base_dims_pix[3] - self.base_original_dims[3])
            self.base_dims_pix[1] = int(self.base_dims_pix[1] + offsetX)
            self.target_dims_pix[1] = int(self.target_dims_pix[1] + offsetX)
            self.base_dims_pix[3] = int(self.base_dims_pix[3] + offsetY)
            self.target_dims_pix[3] = int(self.target_dims_pix[3] + offsetY)
            # Define the number of actual correlation, i.e., the total number of points in row and
            # column, minus the "blank" correlation on the border
            self.nb_corr_col: int = int(np.floor(
                (self.base_dims_pix[2] - self.base_dims_pix[1] + 1 - self.corr_engine.corr_params.window_size[0] - 2 *
                 self.margins[
                     0]) /
                self.corr_engine.corr_params.step[0]) + 1)
            self.nb_corr_row: int = int(np.floor(
                (self.base_dims_pix[4] - self.base_dims_pix[3] + 1 - self.corr_engine.corr_params.window_size[1] - 2 *
                 self.margins[
                     1]) /
                self.corr_engine.corr_params.step[1]) + 1)

            if self.debug:
                logging.info("nbCorrCol:{}, nbCorrRow:{}".format(self.nb_corr_col, self.nb_corr_row))

            #  ;Define the blank border on the right side in column and bottom side in row
            self.border_col_right: int = int(self.tot_col - self.border_col_left - self.nb_corr_col)
            self.border_row_bottom: int = int(self.tot_row - self.border_row_top - self.nb_corr_row)
            if self.debug:
                logging.info(
                    "borderColRight:{}, borderRowBottom:{}".format(self.border_col_right, self.border_row_bottom))

            # Define a "blank" (i.e., invalid) correlation line
            self.output_row_blank = self._blank_array_func(nbVals=self.tot_col)
            ##Define blank column border left and right
            self.output_col_left_blank = self._blank_array_func(nbVals=self.border_col_left)

            self.output_col_right_blank = self._blank_array_func(nbVals=self.border_col_right)

    def tiling(self):
        # TODO refactor this function and create a base class to generate Tiles
        #  see Tile from Jihao python code
        # Get number of pixel in column and row of the file subset to tile
        self.nb_col_img: int = int(self.base_dims_pix[2] - self.base_dims_pix[1] + 1)
        self.nb_row_img: int = int(self.base_dims_pix[4] - self.base_dims_pix[3] + 1)
        if self.debug:
            logging.info("nbColImg: {} || nbRowImg: {}".format(self.nb_col_img, self.nb_row_img))

        # Define number max of lines per tile
        self.max_rows_roi: int = int(
            np.floor(
                (self.tile_size_mb * 8 * 1024 * 1024) / (self.nb_col_img * const.CORRELATION.PIXEL_MEMORY_FOOTPRINT)))
        if self.debug:
            logging.info("maxRowsROI:{}".format(self.max_rows_roi))
        # Define number of correlation column and lines computed for one tile
        if self.max_rows_roi < self.nb_row_img:
            temp = self.max_rows_roi
            self.nb_corr_row_per_roi: int = int((temp - self.win_area_y) / self.corr_engine.corr_params.step[1] + 1)
        else:
            temp = self.nb_row_img
            self.nb_corr_row_per_roi = int((temp - self.win_area_y) / self.corr_engine.corr_params.step[1] + 1)

        self.nb_corr_col_per_roi: int = int(
            (self.nb_col_img - self.win_area_x) / self.corr_engine.corr_params.step[0] + 1)

        # TODO change ROI per tile
        self.nb_roi: int = int(
            (self.nb_row_img - self.win_area_y + self.corr_engine.corr_params.step[1]) / (
                    (self.nb_corr_row_per_roi - 1) * self.corr_engine.corr_params.step[1] + (
                    self.win_area_y - self.corr_engine.corr_params.step[1])))

        if self.nb_roi < 1:
            ## At least one tile even if the ROI is Larger than the image
            self.nb_roi = 1
        if self.debug:
            logging.info("nbROI: {} || nbCorrRowPerROI: {} || nbCorrColPerROI: {}".format(self.nb_roi,
                                                                                          self.nb_corr_row_per_roi,
                                                                                          self.nb_corr_col_per_roi))

        # Define the boundaries of all the tiles but the last one which will have a different size
        self.dims_base_tile = np.zeros((self.nb_roi, 5), dtype=np.int64)
        self.dims_target_tile = np.zeros((self.nb_roi, 5), dtype=np.int64)
        for i in range(self.nb_roi):
            val = int(
                self.base_dims_pix[3] + ((i + 1) * self.nb_corr_row_per_roi - 1) * self.corr_engine.corr_params.step[
                    1] + self.win_area_y - 1)
            self.dims_base_tile[i, :] = [-1,
                                         self.base_dims_pix[1],
                                         self.base_dims_pix[2],
                                         self.base_dims_pix[3] + i * self.nb_corr_row_per_roi *
                                         self.corr_engine.corr_params.step[1],
                                         val]

            self.dims_target_tile[i, :] = [-1,
                                           self.target_dims_pix[1],
                                           self.target_dims_pix[2],
                                           self.target_dims_pix[3] + i * self.nb_corr_row_per_roi *
                                           self.corr_engine.corr_params.step[1],
                                           int(self.target_dims_pix[3] + ((i + 1) * self.nb_corr_row_per_roi - 1) *
                                               self.corr_engine.corr_params.step[1] + self.win_area_y - 1)]

        # Define boundaries of the last tile and the number of correlation column and lines computed for the last tile
        self.nb_rows_left: int = int((self.base_dims_pix[4] - self.dims_base_tile[self.nb_roi - 1, 4] + 1) - 1)
        if self.debug:
            logging.info("nbRowsLeft:{}".format(self.nb_rows_left, "\n"))
        if (self.nb_rows_left >= self.corr_engine.corr_params.step[1]):
            self.nb_corr_row_last_roi: int = int(self.nb_rows_left / self.corr_engine.corr_params.step[1])

            self.dims_base_tile = np.vstack((self.dims_base_tile, np.array(
                [-1, self.base_dims_pix[1], self.base_dims_pix[2],
                 self.base_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi * self.corr_engine.corr_params.step[1],
                 int(self.base_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi * self.corr_engine.corr_params.step[
                     1] + self.win_area_y - 1 + (
                             self.nb_corr_row_last_roi - 1) * self.corr_engine.corr_params.step[1])])))

            self.dims_target_tile = np.vstack((self.dims_target_tile, np.array(
                [-1, self.target_dims_pix[1], self.target_dims_pix[2],
                 self.target_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi * self.corr_engine.corr_params.step[
                     1],
                 int(self.target_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi *
                     self.corr_engine.corr_params.step[
                         1] + self.win_area_y - 1 + (
                             self.nb_corr_row_last_roi - 1) * self.corr_engine.corr_params.step[1])])))
            self.nb_roi = self.nb_roi + 1
        else:
            self.nb_corr_row_last_roi = self.nb_corr_row_per_roi

        return

    def write_blank_pixels(self):
        ## In case of non-gridded correlation,write the top blank correlation lines
        # Define a "blank" (i.e., invalid) correlation line
        temp_add = np.empty((self.border_row_top, self.ew_output.shape[1]))
        temp_add[:] = np.nan
        self.ew_output = np.vstack((temp_add, self.ew_output))
        self.ns_output = np.vstack((temp_add, self.ns_output))
        self.snr_output = np.vstack((temp_add, self.snr_output))

        # In case of non-gridded correlation, write the bottom blank correlation lines
        if self.border_row_bottom != 0:
            temp_add = np.empty((self.border_row_bottom, self.ew_output.shape[1]))
            temp_add[:] = np.nan
            self.ew_output = np.vstack((self.ew_output, temp_add))
            self.ns_output = np.vstack((self.ns_output, temp_add))
            self.snr_output = np.vstack((self.snr_output, temp_add))

        if self.border_col_left != 0:
            ##Define blank column border left and right
            # outputColLeftBlank = BlankArray(nbVals=borderColLeft)
            temp_add = np.empty((self.ew_output.shape[0], self.border_row_bottom))
            temp_add[:] = np.nan
            self.ew_output = np.vstack((temp_add.T, self.ew_output.T)).T
            self.ns_output = np.vstack((temp_add.T, self.ns_output.T)).T
            self.snr_output = np.vstack((temp_add.T, self.snr_output.T)).T
        if self.border_col_right != 0:
            temp_add = np.empty((self.ew_output.shape[0], self.border_col_right))
            temp_add[:] = np.nan
            self.ew_output = np.vstack((self.ew_output.T, temp_add.T)).T
            self.ns_output = np.vstack((self.ns_output.T, temp_add.T)).T
            self.snr_output = np.vstack((self.snr_output.T, temp_add.T)).T

    def set_geo_referencing(self):
        # TODO change this function to a class method
        if all(self.flagList):
            if self.corr_engine.corr_params.grid:
                x_map_origin = self.base_info.x_map_origin + (
                        self.base_dims_pix[1] + self.margins[0] + self.corr_engine.corr_params.window_size[
                    0] / 2) * self.base_info.pixel_width
                y_map_origin = self.base_info.y_map_origin - (
                        self.base_dims_pix[3] + self.margins[1] + self.corr_engine.corr_params.window_size[
                    1] / 2) * np.abs(
                    self.base_info.pixel_height)
            else:
                x_map_origin = self.base_info.x_map_origin + self.base_dims_pix[1] * self.base_info.pixel_width
                y_map_origin = self.base_info.y_map_origin - self.base_dims_pix[3] * np.abs(self.base_info.pixel_height)

            geo_transform = [x_map_origin, self.base_info.pixel_width * self.corr_engine.corr_params.step[0], 0,
                             y_map_origin, 0,
                             -1 * self.base_info.pixel_height * self.corr_engine.corr_params.step[1]]
            if self.debug:
                logging.info("correlation geo. transformation :{}".format(geo_transform))
            return geo_transform, self.base_info.epsg_code
        else:
            logging.warning("=== Pixel Based correlation ===")
            return [0.0, 1.0, 0.0, 0.0, 0.0, -1.0], 4326

    def set_corr_debug(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(os.path.dirname(self.output_corr_path), 'correlation.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )

    @staticmethod
    def make_output_path(path, base_img_path, target_img_path, correlator_name, window_size, step):
        return os.path.join(path,
                            Path(base_img_path).stem + "_VS_" +
                            Path(target_img_path).stem + "_" +
                            correlator_name + "_wz_" +
                            str(window_size) + "_step_" +
                            str(step) + ".tif")


class FreqCorrelator(RawFreqCorr):
    pass


class SpatialCorrelator(RawSpatialCorr):
    pass

class RawNCCCorr(RawSpatialCorr):
    """CPU NCC correlator using skimage.match_template."""
    @classmethod
    def run_correlator(cls, base_array: np.ndarray,
                       target_array: np.ndarray,
                       window_size: list,
                       step: list,
                       search_range: list):
        # window_size is [wx, wy, _, _] in this codebase; use the first two.
        wx, wy = int(window_size[0]), int(window_size[1])
        sx, sy = int(step[0]), int(step[1])
        rx, ry = int(search_range[0]), int(search_range[1])

        H, W = base_array.shape
        # Output grid size mimics Spatial correlator
        out_rows, out_cols = RawSpatialCorr.get_output_dims(
            step_size=[sx, sy],
            input_shape=(H, W),
            window_size=[wx, wy],
            range_size=[rx, ry]
        )
        ew = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
        ns = np.full((out_rows, out_cols), np.nan, dtype=np.float32)
        snr = np.full((out_rows, out_cols), np.nan, dtype=np.float32)

        r = 0
        for y0 in range(ry, H - (wy + ry) + 1, sy):
            c = 0
            for x0 in range(rx, W - (wx + rx) + 1, sx):
                # Base patch (template) centered at (x0 + wx/2, y0 + wy/2)
                tpl = base_array[y0:y0 + wy, x0:x0 + wx]

                # Search window in target around the same nominal location ± range
                ys = max(0, y0 - ry)
                xs = max(0, x0 - rx)
                ye = min(H, y0 + wy + ry)
                xe = min(W, x0 + wx + rx)
                search = target_array[ys:ye, xs:xe]

                if search.shape[0] < wy or search.shape[1] < wx:
                    c += 1
                    continue

                # Normalized cross-correlation; returns a response map
                resp = match_template(search, tpl, pad_input=False, mode="constant")
                # Best match index:
                ij = np.unravel_index(np.argmax(resp), resp.shape)
                dy_local, dx_local = int(ij[0]), int(ij[1])
                peak = float(resp[ij])

                # Convert to displacement relative to template's top-left
                # (dx,dy) positive right/down in pixels
                #dx = (dx_local + wx//2) - rx
                #dy = (dy_local + wy//2) - ry

                #dx = dx_local - rx  
                #dy = dy_local - ry 

                #dx = (xs + dx_local) - x0 
                #dy = (ys + dy_local) - y0
                dx = (xs + dx_local + wx//2) - (x0 + wx//2)
                dy = (ys + dy_local + wy//2) - (y0 + wy//2)

                # Save: East/West (+E ~ +x), North/South (+N is -dy if y increases downward)
                ew[r, c] = float(dx)
                ns[r, c] = float(dy)  # flip sign to keep +north consistent
                snr[r, c] = peak
                c += 1
            r += 1

        return ew, ns, snr


class NCCCorrelator(RawNCCCorr):
    pass

def _swt_feature_stack(img: np.ndarray,
                       wavelet='db2',
                       levels=3,
                       pad_mode='edge',
                       trim_approx=True,
                       norm=True):
    """Return list of SWT feature maps (|details|) per level, each HxW.
       - Pads to multiples of 2**levels so swt2 won't error.
       - Works with both trim_approx=True (details only) and False (approx+details).
    """
    img = img.astype(np.float32, copy=False)
    H, W = img.shape

    # Cap levels if tile is too small
    maxL = int(math.floor(math.log2(max(1, min(H, W)))))
    levels = max(1, min(levels, maxL))

    # Pad to multiples of 2**levels
    blk = 1 << levels
    pad_y = (-H) % blk
    pad_x = (-W) % blk
    if pad_y or pad_x:
        img_pad = np.pad(img, ((0, pad_y), (0, pad_x)), mode=pad_mode)
    else:
        img_pad = img

    coeffs = pywt.swt2(img_pad, wavelet=wavelet, level=levels,
                       trim_approx=trim_approx, norm=norm)

    feats = []
    
    if trim_approx:
        # When trim_approx=True, PyWavelets returns [approx, (cH1,cV1,cD1), (cH2,cV2,cD2), ...]
        # We need to skip the first element (approximation) and process the detail tuples
        detail_coeffs = coeffs[1:]  # Skip approximation coefficients
        
        for level_idx, coeff_item in enumerate(detail_coeffs):
            if isinstance(coeff_item, tuple) and len(coeff_item) == 3:
                cH, cV, cD = coeff_item
                
                # Ensure they're 2D and have reasonable dimensions
                if (cH.ndim == 2 and cV.ndim == 2 and cD.ndim == 2 and 
                    cH.shape == cV.shape == cD.shape):
                    mag = np.sqrt(cH*cH + cV*cV + cD*cD).astype(np.float32)
                    # Crop back to original size (coefficients may be larger due to padding)
                    mag_cropped = mag[:H, :W] if mag.shape[0] >= H and mag.shape[1] >= W else mag
                    feats.append(mag_cropped)
    else:
        # When trim_approx=False, format is [(cA1, (cH1, cV1, cD1)), (cA2, (cH2, cV2, cD2)), ...]
        for level_idx, coeff_item in enumerate(coeffs):
            if isinstance(coeff_item, tuple) and len(coeff_item) == 2:
                cA, details = coeff_item
                if isinstance(details, tuple) and len(details) == 3:
                    cH, cV, cD = details
                    
                    # Ensure they're 2D and have reasonable dimensions
                    if (cH.ndim == 2 and cV.ndim == 2 and cD.ndim == 2 and 
                        cH.shape == cV.shape == cD.shape):
                        mag = np.sqrt(cH*cH + cV*cV + cD*cD).astype(np.float32)
                        # Crop back to original size
                        mag_cropped = mag[:H, :W] if mag.shape[0] >= H and mag.shape[1] >= W else mag
                        feats.append(mag_cropped)
    
    return feats
    
def _dtcwt_feature_stack(img: np.ndarray, levels=4):
    """
    Dual-tree complex wavelet transform for better directional selectivity.
    Returns 6 directional subbands per level (total 6*levels features).
    """
    img = img.astype(np.float32, copy=False)
    H, W = img.shape
    
    # Initialize transform
    transform = dtcwt.Transform2d()
    
    # Forward transform
    try:
        coeffs = transform.forward(img, nlevels=levels)
    except ValueError as e:
        # Handle size constraints - DTCWT needs certain dimensions
        pad_h = ((H + 15) // 16) * 16 - H  # Pad to multiple of 16
        pad_w = ((W + 15) // 16) * 16 - W
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
        coeffs = transform.forward(img_padded, nlevels=levels)
    
    feats = []
    # Extract magnitude of complex coefficients at each level
    for level in range(levels):
        highpasses = coeffs.highpasses[level]
        h_level, w_level = highpasses.shape[:2]
        
        # Ensure we can crop back to original size
        if h_level < H or w_level < W:
            # If subband is smaller than original, pad it
            for direction in range(6):
                mag = np.abs(highpasses[:, :, direction]).astype(np.float32)
                # Pad to at least original size
                if mag.shape[0] < H or mag.shape[1] < W:
                    pad_h = max(0, H - mag.shape[0])
                    pad_w = max(0, W - mag.shape[1])
                    mag = np.pad(mag, ((0, pad_h), (0, pad_w)), mode='edge')
                # Crop to exact original size
                mag_cropped = mag[:H, :W]
                feats.append(mag_cropped)
        else:
            # Normal case - crop to original size
            for direction in range(6):
                mag = np.abs(highpasses[:, :, direction]).astype(np.float32)
                mag_cropped = mag[:H, :W]
                feats.append(mag_cropped)
    
    # Verify all features have the same shape
    for i, feat in enumerate(feats):
        if feat.shape != (H, W):
            # Force resize if still wrong shape
            feats[i] = np.resize(feat, (H, W))
    
    return feats


def _parabolic_subpixel(resp: np.ndarray, v0: int, u0: int):
    H, W = resp.shape
    r00 = resp[v0, u0]
    rxm = resp[v0, max(u0-1,0)]; rxp = resp[v0, min(u0+1,W-1)]
    rym = resp[max(v0-1,0), u0]; ryp = resp[min(v0+1,H-1), u0]
    dx_den = (rxm - 2*r00 + rxp); dy_den = (rym - 2*r00 + ryp)
    dx = 0.5*(rxm - rxp) / (dx_den if abs(dx_den) > 1e-9 else (1e-9 if dx_den >= 0 else -1e-9))
    dy = 0.5*(rym - ryp) / (dy_den if abs(dy_den) > 1e-9 else (1e-9 if dy_den >= 0 else -1e-9))
    return float(dx), float(dy)

class WaveletNCCCorr(RawSpatialCorr):
    """Spatial correlator: NCC on SWT magnitudes, fused across levels."""
    def __init__(self, window_size=None, steps=None, search_range=None, grid=True,
                 wavelet='db2', levels=3, min_tpl_var=1e-4, min_peak=0.5):
        super().__init__(window_size, steps, search_range, grid)
        self.wavelet = wavelet
        self.levels = levels
        self.min_tpl_var = min_tpl_var
        self.min_peak = min_peak

    @classmethod
    def run_correlator(cls, base_array: np.ndarray, target_array: np.ndarray,
                       window_size: list, step: list, search_range: list,
                       wavelet='db2', levels=3, min_tpl_var=1e-4, min_peak=0.5):
        wx, wy = int(window_size[0]), int(window_size[1])
        sx, sy = int(step[0]), int(step[1])
        rx, ry = int(search_range[0]), int(search_range[1])

        H, W = base_array.shape
        out_rows, out_cols = RawSpatialCorr.get_output_dims(
            step_size=[sx, sy], input_shape=(H, W),
            window_size=[wx, wy], range_size=[rx, ry]
        )
        ew = np.full((out_rows, out_cols), np.nan, np.float32)
        ns = np.full((out_rows, out_cols), np.nan, np.float32)
        q  = np.full((out_rows, out_cols), np.nan, np.float32)

        # Precompute SWT features once per tile
        F_base   = _swt_feature_stack(base_array, wavelet=wavelet, levels=levels)
        F_target = _swt_feature_stack(target_array, wavelet=wavelet, levels=levels)

        r = 0
        for y0 in range(ry, H - (wy + ry) + 1, sy):
            c = 0
            for x0 in range(rx, W - (wx + rx) + 1, sx):
                ys, xs = max(0, y0 - ry), max(0, x0 - rx)
                ye, xe = min(H, y0 + wy + ry), min(W, x0 + wx + rx)
                if (ye-ys) < wy or (xe-xs) < wx:
                    c += 1; continue

                resp_fused, wsum, valid = None, 0.0, False
                for fb, ft in zip(F_base, F_target):
                    tpl = fb[y0:y0+wy, x0:x0+wx]
                    if np.nanvar(tpl) < min_tpl_var:
                        continue
                    search = ft[ys:ye, xs:xe]
                    resp = match_template(search, tpl, pad_input=False, mode="constant")
                    w = float(max(1e-6, np.mean(tpl)))
                    resp_fused = (resp*w) if resp_fused is None else (resp_fused + resp*w)
                    wsum += w; valid = True

                if not valid or wsum <= 0:
                    c += 1; continue

                resp_fused /= wsum
                v0, u0 = np.unravel_index(np.nanargmax(resp_fused), resp_fused.shape)
                peak = float(resp_fused[v0, u0])
                if not np.isfinite(peak) or peak < min_peak:
                    c += 1; continue

                dx_sub, dy_sub = _parabolic_subpixel(resp_fused, v0, u0)
                dx = (xs + u0 + wx//2) - (x0 + wx//2) + dx_sub
                dy = (ys + v0 + wy//2) - (y0 + wy//2) + dy_sub

                ew[r, c] = float(dx)
                ns[r, c] = float(dy)   # +North = up (row+ is down)
                q[r, c]  = peak
                c += 1
            r += 1
        return ew, ns, q

class WaveletNCCCorrelator(WaveletNCCCorr):
    pass

class DTCWTNCCCorr(RawSpatialCorr):
    """Spatial correlator using Dual-Tree Complex Wavelet Transform."""
    
    def __init__(self, window_size=None, steps=None, search_range=None, grid=True,
                 levels=4, min_tpl_var=1e-4, min_peak=0.5):
        super().__init__(window_size, steps, search_range, grid)
        self.levels = levels
        self.min_tpl_var = min_tpl_var
        self.min_peak = min_peak
    
    @classmethod
    def run_correlator(cls, base_array: np.ndarray, target_array: np.ndarray,
                       window_size: list, step: list, search_range: list,
                       levels=4, min_tpl_var=1e-4, min_peak=0.5):
        wx, wy = int(window_size[0]), int(window_size[1])
        sx, sy = int(step[0]), int(step[1])
        rx, ry = int(search_range[0]), int(search_range[1])
        
        H, W = base_array.shape
        out_rows, out_cols = RawSpatialCorr.get_output_dims(
            step_size=[sx, sy], input_shape=(H, W),
            window_size=[wx, wy], range_size=[rx, ry]
        )
        ew = np.full((out_rows, out_cols), np.nan, np.float32)
        ns = np.full((out_rows, out_cols), np.nan, np.float32)
        q  = np.full((out_rows, out_cols), np.nan, np.float32)
        
        # Compute DTCWT features once per tile
        F_base = _dtcwt_feature_stack(base_array, levels=levels)
        F_target = _dtcwt_feature_stack(target_array, levels=levels)
        
        r = 0
        for y0 in range(ry, H - (wy + ry) + 1, sy):
            c = 0
            for x0 in range(rx, W - (wx + rx) + 1, sx):
                ys, xs = max(0, y0 - ry), max(0, x0 - rx)
                ye, xe = min(H, y0 + wy + ry), min(W, x0 + wx + rx)
                if (ye-ys) < wy or (xe-xs) < wx:
                    c += 1; continue
                
                resp_fused, wsum, valid = None, 0.0, False
                for fb, ft in zip(F_base, F_target):
                    tpl = fb[y0:y0+wy, x0:x0+wx]
                    if np.nanvar(tpl) < min_tpl_var:
                        continue
                    search = ft[ys:ye, xs:xe]
                    resp = match_template(search, tpl, pad_input=False, mode="constant")
                    w = float(max(1e-6, np.mean(tpl)))
                    resp_fused = (resp*w) if resp_fused is None else (resp_fused + resp*w)
                    wsum += w; valid = True
                
                if not valid or wsum <= 0:
                    c += 1; continue
                
                resp_fused /= wsum
                v0, u0 = np.unravel_index(np.nanargmax(resp_fused), resp_fused.shape)
                peak = float(resp_fused[v0, u0])
                if not np.isfinite(peak) or peak < min_peak:
                    c += 1; continue
                
                dx_sub, dy_sub = _parabolic_subpixel(resp_fused, v0, u0)
                dx = (xs + u0 + wx//2) - (x0 + wx//2) + dx_sub
                dy = (ys + v0 + wy//2) - (y0 + wy//2) + dy_sub
                
                ew[r, c] = float(dx)
                ns[r, c] = float(dy)
                q[r, c] = peak
                c += 1
            r += 1
        return ew, ns, q

class DTCWTNCCCorrelator(DTCWTNCCCorr):
    pass