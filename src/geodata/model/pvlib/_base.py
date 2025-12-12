# Copyright 2016-2017 Gorm Andresen (Aarhus University), Jonas Hoersch (FIAS), Tom Brown (FIAS)
# Copyright 2020 Michael Davidson (UCSD), William Honaker, Jiahe Feng (UCSD), Yuanbo Shi
# Copyright 2023-2024 Xiqiang Liu, 2025 Keyu Long

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
GEODATA

Geospatial Data Collection and "Pre-Analysis" Tools

TODO: Documentation here

"""

import time
import pandas as pd
import numpy as np
import xarray as xr
from timezonefinder import TimezoneFinder
from pvlib import pvsystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.atmosphere import gueymard94_pw
from pvlib.solarposition import get_solarposition

from .._base import BaseModel
from geodata.logging import logger


class ModelChainConfig:
    """
    Defines pvlib ModelChain parameters as a class that
    can be passed to one or more instances of pvlib_model().
    Allows user to reuse a common set of ModelChain parameters across multiple
    PVSystems or even multiple cutouts.  

    Parameters
    ----------
    clearsky_model : string, default 'ineichen'
        Specifies the clear-sky model. Passed to location.get_clearsky. 
        Only used when DNI is not found in the weather inputs.
    transposition_model : string, default 'haydavies'
        Specifies the transposition model. Passed to system.get_irradiance.
    solar_position_method : string, default 'nrel_numpy'
        Specifies the method for calculating solar positions. Passed to location.get_solarposition.
    airmass_model : string, default 'kastenyoung1989'
        Specifies the airmass model. Passed to location.get_airmass.
    dc_model : string or function, optional
        Specifies the DC model. Valid strings are 'sapm', 'desoto', 'cec', 'pvsyst', 'pvwatts'. 
        If not specified, the model will be inferred from the parameters of system.arrays[i].module_parameters. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    ac_model : string or function, optional
        Specifies the AC model. Valid strings are 'sandia', 'adr', 'pvwatts'. 
        If not specified, the model will be inferred from the parameters of system.inverter_parameters. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    aoi_model : string or function, optional
        Specifies the angle of incidence (AOI) model. Valid strings are 'physical', 'ashrae', 'sapm', 'martin_ruiz', 
        'interp', 'no_loss'. If not specified, the model will be inferred from the parameters of 
        system.arrays[i].module_parameters. A user-defined function may also be provided, 
        with the ModelChain instance passed as the first argument.
    spectral_model : string or function, optional
        Specifies the spectral model. Valid strings are 'sapm', 'first_solar', 'no_loss'. 
        If not specified, the model will be inferred from the parameters of system.arrays[i].module_parameters. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    temperature_model : string or function, optional
        Specifies the temperature model. Valid strings are 'sapm', 'pvsyst', 'faiman', 'fuentes', 'noct_sam'. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    dc_ohmic_model : string or function, default 'no_loss'
        Specifies the DC ohmic loss model. Valid strings are 'dc_ohms_from_percent', 'no_loss'. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    losses_model : string or function, default 'no_loss'
        Specifies the losses model. Valid strings are 'pvwatts', 'no_loss'. 
        A user-defined function may also be provided, with the ModelChain instance passed as the first argument.
    name : string, optional
        Specifies the name of the ModelChain instance.
    
    For full documentation, see:
    - pvlib.modelchain.ModelChain(): 
        https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.html

    """
    def __init__(
        self,
        clearsky_model='ineichen',
        transposition_model='haydavies',
        solar_position_method='nrel_numpy',
        airmass_model='kastenyoung1989',
        dc_model=None,
        ac_model=None,
        aoi_model=None,
        spectral_model=None,
        temperature_model=None,
        dc_ohmic_model='no_loss',
        losses_model='no_loss',
        name=None
    ):
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model
        self.dc_model = dc_model
        self.ac_model = ac_model
        self.aoi_model = aoi_model
        self.spectral_model = spectral_model
        self.temperature_model = temperature_model
        self.dc_ohmic_model = dc_ohmic_model
        self.losses_model = losses_model
        self.name = name

    def model_chain_to_kwargs(self):
        return self.__dict__
    

class PVLib(BaseModel):
    """The PVLib model"""

    type: str = "pvlib"

    SUPPORTED_WEATHER_DATA_CONFIGS = ("wind_solar_hourly",)

    @property
    def prepared(self) -> bool:
        """This model does not need to be prepared"""
        return True
    
    def prepare(self, force: bool = False):
        """Skip preparation - this model doesn't need it."""
        logger.info("This model does not require preparation. Skipping.")
        return

    def init_model_config(
        self,
        clearsky_model='ineichen',
        transposition_model='haydavies',
        solar_position_method='nrel_numpy',
        airmass_model='kastenyoung1989',
        dc_model=None,
        ac_model=None,
        aoi_model=None,
        spectral_model=None,
        temperature_model=None,
        dc_ohmic_model='no_loss',
        losses_model='no_loss',
        name=None
    ):
        self.config = ModelChainConfig(
            clearsky_model= clearsky_model,
            transposition_model= transposition_model, 
            solar_position_method= solar_position_method,
            airmass_model= airmass_model,
            dc_model= dc_model,
            ac_model= ac_model, 
            aoi_model= aoi_model,
            spectral_model= spectral_model,
            temperature_model= temperature_model,
            dc_ohmic_model= dc_ohmic_model,
            losses_model= losses_model,
            name= name
        )

    def _estimate_dataset(self, params: xr.Dataset, **kwargs) -> xr.Dataset | xr.DataArray:  # type: ignore[override]
        """Estimate PV output from prepared dataset.
        
        Args:
            params: Dataset (already filtered by years/months/xs/ys from BaseModel)
            **kwargs: Additional parameters (not used currently, but available)
        
        Returns:
            Dataset with AC power and PV capacity (returns Dataset, but BaseModel expects DataArray)
        """
        if self.pv_system is None:
            raise ValueError("pv_system is not initialized. Call init_pv_system() first.")
        if self.config is None:
            raise ValueError("model_config is not initialized. Call init_model_config() first.")
        
        # Call the core pvlib model function
        result = self._pvlib_model(
            params,  # Dataset instead of cutout
            self.pv_system,
            self.config
        )
        # Return as Dataset (will be converted if needed by caller)
        # Note: BaseModel expects DataArray, but PVLib naturally returns Dataset with 'ac' and 'pv'
        return result

    def estimate(self,
        years: slice | None = None,
        months: slice | None = None,
        xs: slice | None = None,
        ys: slice | None = None,
        **kwargs,
    ) -> xr.Dataset | xr.DataArray:  # type: ignore[override]
        """Estimate PV output with validation.
        
        Args:
            years: Year range (slice)
            months: Month range (slice)
            xs: X-coordinate range (slice)
            ys: Y-coordinate range (slice)
            **kwargs: Additional parameters
        
        Returns:
            Dataset with AC power and PV capacity
        """
        # Validate required components
        if self.pv_system is None:
            raise ValueError("pv_system is not initialized. Call init_pv_system() first.")
        if self.config is None:
            raise ValueError("model_config is not initialized. Call init_model_config() first.")
        
        # Override estimate to load from raw files (ref_files) instead of prepared files
        # since PVLib doesn't use the preparation step
        if years is None and months is None:
            results = self.flattened_results
        elif months is None:
            if years is None:
                years = slice(self.source.years.start, self.source.years.stop)
            results = self.get_result_year_month(years, slice(1, 13))
        else:
            if years is None:
                years = slice(self.source.years.start, self.source.years.stop)
            results = self.get_result_year_month(years, months)

        # Use ref_files (raw dataset files) instead of files (prepared files)
        files = sum([result.ref_files for result in results], [])
        from .._base import _get_xr_engine, _should_use_parallel_reading
        
        engine = "h5netcdf"
        parallel = _should_use_parallel_reading()
        logger.info(
            f"estimate: Opening {len(files)} files with engine={engine}, parallel={parallel}"
        )
        params = xr.open_mfdataset(files, engine=engine, parallel=parallel)

        if xs is not None:
            params = params.sel(x=xs)
        if ys is not None:
            params = params.sel(y=ys)

        # Debug: Print available variables (helpful for debugging)
        logger.info(f"Available variables in dataset: {list(params.data_vars.keys())}")
        logger.debug(f"Available coordinates: {list(params.coords.keys())}")
        logger.debug(f"Dataset dimensions: {dict(params.dims)}")
        
        output = self._estimate_dataset(params, **kwargs)
        params.close()
        return output
    
    def retrieve_sam(self, samfile, path=None):
        """
        Wrapper for pvlib.pvsystem.retrieve_sam(). Retrieves latest module 
        and inverter info from a file bundled with pvlib, a path or a 
        URL (like SAM’s website), and returns it as a Pandas DataFrame.

        Supported databases:
        - CEC module database
        - Sandia Module database
        - CEC Inverter database
        - Anton Driesse Inverter database

        Parameters
        ----------
        name : string
                Use one of the following strings to retrieve a database bundled with pvlib:
                    - ’CECMod’ - returns the CEC module database
                    - ’CECInverter’ - returns the CEC Inverter database
                    - ’SandiaInverter’ - returns the CEC Inverter database 
                        (CEC is only current inverter db available; tag kept for backwards compatibility)
                    - ’SandiaMod’ - returns the Sandia Module database
                    - ’ADRInverter’ - returns the ADR Inverter database
        
        Optional Parameters
        ----------
        path : string
                Path to a CSV file or a URL.

        Returns: DataFrame

        See also:
            - pvlib.pvsystem.retrieve_sam(): 
                https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.retrieve_sam.html

        """
        return pvsystem.retrieve_sam(name=samfile, path=path)

    def init_pv_system(self, *args, **kwargs):
        """
        Wrapper for pvlib.pvsystem.PVSystem().
        The PVSystem class defines a standard set of PV system attributes
        and modeling functions. This class describes the collection and 
        interactions of PV system components rather than an installed system
        on the ground. It is typically used in combination with Location 
        and ModelChain objects.

        The class supports basic system topologies consisting of:
            - N total modules arranged in series (modules_per_string=N, strings_per_inverter=1).
            - M total modules arranged in parallel (modules_per_string=1, strings_per_inverter=M).
            - NxM total modules arranged in M strings of N modules each 
            (modules_per_string=N, strings_per_inverter=M).

        For full documentation, see: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.PVSystem.html

        Parameters
        ----------
        arrays : array (optional)
                An Array or list of arrays that are part of the system. 
                See pvlib documentation for full description.
        surface_tilt : float
                Surface tilt angles in decimal degrees. The tilt angle is 
                defined as degrees from horizontal (e.g. surface facing up = 0, 
                surface facing horizon = 90).
        surface_azimuth : float
                Azimuth angle of the module surface. North=0, East=90, South=180, West=270.
        albedo : float
                Ground surface albedo. If not supplied, then surface_type is used to look up 
                a value in pvlib.albedo.SURFACE_ALBEDOS. If surface_type is also not supplied 
                then a ground surface albedo of 0.25 is used.
        surface_type : string
                The ground surface type. See pvlib.albedo.SURFACE_ALBEDOS for valid values.
        module : string
                The model name of the modules. May be used to look up the module_parameters dictionary via some other method.
        module_type : string 
                Describes the module’s construction. Valid strings are ‘glass_polymer’ and ‘glass_glass’. 
                Used for cell and module temperature calculations.
        module_parameters : dict
                Module parameters as defined by the SAPM, CEC, or other.
        temperature_model_parameters : dict
                Temperature model parameters as required by one of the models in pvlib.temperature (excluding poa_global, temp_air and wind_speed).
        modules_per_string : int, float
                See system topology discussion above.
        strings_per_inverter : int, float 
                See system topology discussion above.
        inverter : string 
                The model name of the inverters. May be used to look up the inverter_parameters dictionary via some other method.
        inverter_parameters : dict
                Inverter parameters as defined by the SAPM, CEC, or other.
        racking_model : string 
                Valid strings are ‘open_rack’, ‘close_mount’, and ‘insulated_back’. 
                Used to identify a parameter set for the SAPM cell temperature model.
        losses_parameters : dict 
                Losses parameters as defined by PVWatts or other.    
        name : string (optional)
        
        """
        self.pv_system = pvsystem.PVSystem(*args, **kwargs)
    
    def _pvlib_model(
            self,
            ds: xr.Dataset,  # Changed from cutout to Dataset
            system,
            model_chain_config,
            vars = [
                'influx_diffuse', 
                'influx_direct', 
                #'dewpoint_temperature',
                'd2m', 
                'temperature', 
                'wnd100m'
            ]
        ):

        """
        Applies a `pvlib` model using :code:`pvlib.modelchain.ModelChain()` across all unique coordinates 
        represented in a `geodata` cutout. This function prepares input weather data, initializes the 
        `pvlib` model, and runs simulations for each set of coordinates, outputting an xarray dataset 
        containing all simulation results.

        Requires a cutout with the following variables:

        - **influx_diffuse** (*float*) - Diffuse horizontal irradiance.  
        - **influx_direct** (*float*) - Direct normal irradiance.  
        - **dewpoint_temperature** (*float*) - Dewpoint temperature in Celsius.  
        - **temperature** (*float*) - Air temperature in Celsius.  
        - **wnd100m** (*float*) - Wind speed at 100m.  

        Outputs an `xarray.Dataset` containing:

        - **ac** (*float*) - AC photovoltaic output (W).
        - **pv** (*float*) - Photovoltaic capacity.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset generated by the `geodata` library, based on the ERA5 dataset.  
            Must contain the required meteorological variables. Already filtered by years/months/xs/ys.
        system : pvlib **PVSystem** class
            The photovoltaic system to be simulated.  Generated by :code:`geodata.pvlib.init_pv_system()`
        model_chain_config : `ModelChainConfig`
            Configuration object for :code:`pvlib.modelchain.ModelChain()` with model parameters.
        vars : list of str, optional
            List of variable names required for simulation. Defaults to:
            ['influx_diffuse', 'influx_direct', 'd2m', 'temperature', 'wnd100m'].

        Returns
        -------
        xr.Dataset
            Dataset containing ac power output and pv capacity across all coordinates in the dataset.

        """
        ptc = system.arrays[0].module_parameters['PTC']
        n_mods = system.arrays[0].modules_per_string

        weather_data = self._prepare_pvlib_ds(ds, *vars).to_dataframe()
        unique_coords = weather_data.index.droplevel('time').drop_duplicates()
        
        num_coords = len(unique_coords)
        logger.debug(f"Starting pvlib model computation for {num_coords} coordinates")
        start_time = time.time()
        
        # Log progress every N coordinates or every 10% (whichever is more frequent)
        log_interval = max(1, min(1000, num_coords // 10))
        last_log_time = start_time
        
        coord_subsets = []
        for idx, (y, x) in enumerate(unique_coords, 1):
            subset = weather_data.loc[(slice(None), y, x), :].reset_index(['x', 'y'])
            # Normalize longitude to [-180, 180] range for TimezoneFinder
            # which expects longitude in this range. Handle both [0, 360] and [-180, 180] formats
            if x > 180:
                x_normalized = x - 360.0
            elif x < -180:
                x_normalized = x + 360.0
            else:
                x_normalized = x
            tz_str = TimezoneFinder().timezone_at(lat=y, lng=x_normalized)
            if tz_str is None:
                tz_str = "UTC"  # Default to UTC if timezone not found
            location = Location(latitude=y, longitude=x_normalized, tz=tz_str)
            
            mc = ModelChain(
                system, 
                location, 
                **model_chain_config.model_chain_to_kwargs()
            )
            mc.run_model(subset)
            
            subset['ac'] = mc.results.ac
            subset.loc[subset['ac'] < 0, 'ac'] = 0
            subset['pv'] = subset['ac'] / (ptc * n_mods)

            coord_subsets.append(subset)
            
            # Log progress periodically if debug level is enabled
            if idx % log_interval == 0 or idx == num_coords:
                elapsed = time.time() - start_time
                elapsed_since_last = time.time() - last_log_time
                rate = log_interval / elapsed_since_last if elapsed_since_last > 0 else 0
                percent = 100 * idx / num_coords
                logger.debug(
                    f"Progress: {idx}/{num_coords} coordinates ({percent:.1f}%) "
                    f"processed in {elapsed:.1f}s (rate: {rate:.1f} coords/s)"
                )
                last_log_time = time.time()
        
        elapsed_time = time.time() - start_time
        logger.debug(
            f"Completed pvlib model computation for {num_coords} coordinates "
            f"in {elapsed_time:.2f}s (avg {elapsed_time/num_coords:.3f}s per coordinate)"
        )

        weather_data_final = pd.concat(coord_subsets)

        return xr.Dataset.from_dataframe(weather_data_final)

    def _prepare_dataset(self, source: xr.Dataset) -> xr.Dataset:
        """This will never be called, but must be implemented (abstract method)."""
        raise NotImplementedError("This model does not use _prepare_dataset")
    

    def _prepare_pvlib_ds(self, ds: xr.Dataset, *varnames) -> xr.Dataset:
            """
            Prepares an `xarray.Dataset` from a geodata dataset for use in model simulations using `pvlib`.
            This function extracts specified variables from the dataset, calculates additional parameters 
            like global horizontal irradiance (GHI), precipitable water, and solar position, and renames fields to 
            align with expected inputs.

            This function handles both raw ERA5 variables and preprocessed variables. If raw variables are detected,
            it applies the preprocessing transformations.

            Outputs an `xarray.Dataset` with the following variables:

            - **dhi** (*float*) - Diffuse horizontal irradiance.  
            - **dni** (*float*) - Direct normal irradiance.  
            - **ghi** (*float*) - Global horizontal irradiance (calculated via :code:`_calculate_ghi()`).  
            - **temp_air** (*float*) - Air temperature in Celsius.  
            - **wind_speed** (*float*) - Wind speed at 100m.  
            - **precipitable_water** (*float*) - Precipitable water (calculated via :code:`_calculate_precipitable_water()`).

            Parameters
            ----------
            ds : xarray.Dataset
                Dataset generated by `geodata` library. Can contain either raw ERA5 variables or preprocessed variables.
            varnames : string
                String values representing names of required variables to extract (optional).

            Returns
            -------
            weather_data : `xarray.Dataset`
                Dataset containing necessary variables to run `pvlib` model simulations.

            """
            # Check if dataset is preprocessed - datasets should be preprocessed
            # during download, not during model estimation. This avoids parallel
            # reading file handle issues.
            try:
                from ...datasets.era5.wind_solar._base import ERA5WindSolarBaseDataset
                if not ERA5WindSolarBaseDataset.is_preprocessed(ds):
                    raise ValueError(
                        "Dataset is not preprocessed. Please ensure the dataset was "
                        "downloaded with a recent version of geodata that performs "
                        "preprocessing during download. If you have existing raw data, "
                        "you may need to re-download it or manually preprocess it."
                    )
                logger.debug("Dataset is preprocessed. Available variables: %s", list(ds.data_vars.keys()))
            except ImportError:
                logger.warning("Could not import ERA5WindSolarBaseDataset. Assuming dataset is preprocessed.")
                # If import fails, assume dataset is already preprocessed
            
            # Extract only needed variables if specified
            if varnames:
                # Check which variables are actually available
                available_vars = [v for v in varnames if v in ds.data_vars]
                missing_vars = [v for v in varnames if v not in ds.data_vars]
                if missing_vars:
                    logger.warning(f"Missing variables: {missing_vars}. Available: {list(ds.data_vars.keys())}")
                if available_vars:
                    ds = ds[available_vars]
                else:
                    logger.error(f"None of the requested variables {varnames} are available in dataset")
                    raise KeyError(f"None of the requested variables {varnames} are available. Available variables: {list(ds.data_vars.keys())}")

            temperature_celsius = self._convert_celsius(ds.temperature)

            relative_humidity = self._calculate_relative_humidity(
                temperature_celsius,
                #self._convert_celsius(ds.dewpoint_temperature),
                self._convert_celsius(ds.d2m),
            )

            precipitable_water = self._calculate_precipitable_water(
                temperature_celsius,
                relative_humidity
            )

            sp = self._calculate_pvlib_solarposition(ds)
            ghi = self._calculate_ghi(ds, sp['zenith'])

            ds = (
                ds
                .assign(
                    ghi=ghi,
                    temperature=temperature_celsius,
                    precipitable_water=precipitable_water
                )
                .rename({
                    'influx_diffuse': 'dhi',
                    'influx_direct': 'dni',
                    'temperature': 'temp_air',
                    'wnd100m': 'wind_speed'
                })
            )

            return ds[[
                "dhi", 
                "dni",
                "ghi", 
                "temp_air", 
                "wind_speed", 
                "precipitable_water"
            ]]

    # --- Static Physics Helpers (Pure Math) ---
    @staticmethod
    def _calculate_pvlib_solarposition(ds):
        """
        Wrapper for :code:`pvlib.solarposition.get_solarposition()`.  
        Allows for vectorized calculation of solar position across an xarray dataset.
        The solar zenith angle is a required input for :code:`_calculate_ghi()`.

        For full documentation on how :code:`pvlib.solarposition.get_solarposition()` calculates precipitable water,
        see: `the pvlib API reference for pvlib.solarposition.get_solarposition() <https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.solarposition.get_solarposition.html>`.

        Parameters
        ----------
        ds : xarray dataset
            An xarray dataset containing series for both influx diffuse (dhi) and influx direct (dni).
        zenith : numeric
            Zenith angle of the sun in degrees, as calculated by :code:`_calculate_pvlib_solarposition()`.

        Returns
        -------
        solarposition : dataframe
            Dataframe containing solar zenith angle for a given time and set of coordinates.
        """
        nt, ny, nx = ds.sizes['time'], ds.sizes['y'], ds.sizes['x']
        time_expanded = np.broadcast_to(ds.time.values[:, None, None], (nt, ny, nx)).ravel()
        yy, xx = np.meshgrid(ds.y, ds.x, indexing="ij")
        x_expanded = np.tile(xx.ravel(), nt)
        y_expanded = np.tile(yy.ravel(), nt)
        solarposition_df = get_solarposition(time_expanded, y_expanded, x_expanded)
        # Create MultiIndex and assign to DataFrame index
        multi_index = pd.MultiIndex.from_arrays([time_expanded, y_expanded, x_expanded], names=['time', 'y', 'x'])
        if isinstance(solarposition_df, pd.DataFrame):
            solarposition_df = solarposition_df.set_index(multi_index)
        else:
            # If it's a Series or other type, convert to DataFrame first
            solarposition_df = pd.DataFrame(solarposition_df)
            solarposition_df.index = multi_index
        return solarposition_df
    
    @staticmethod
    def _calculate_ghi(ds, zenith):
        """
        Calculates global horizontal irradiance (ghi) from data arrays representing influx diffuse (dhi) and influx direct (dni)
        Negative values are clipped.  Calculated using the formula:

        .. math::

        GHI = DHI + DNI * cos(Z)

        where Z representst the solar zenith as calculated by :code:`_calculate_pvlib_solarposition()`.

        Parameters
        ----------
        ds : xarray dataset
            An xarray dataset containing series for both influx diffuse (dhi) and influx direct (dni).
        zenith : numeric
            Zenith angle of the sun in degrees, as calculated by :code:`_calculate_pvlib_solarposition()`.

        Returns
        -------
        ghi : numeric
            Global horizontal irradiance (ghi) [W m**-2].

        """
        dhi = ds.influx_diffuse.values.ravel()
        dni = ds.influx_direct.values.ravel()
        ghi = np.clip(
            dhi + dni * np.cos(zenith),
            0,
            #np.Inf
            np.inf # `np.Inf` was removed in the NumPy 2.0 release.
        )

        reshaped_ghi = ghi.values.reshape(
            ds.sizes['time'], 
            ds.sizes['y'], 
            ds.sizes['x']
        )
        
        ghi = xr.DataArray(
            reshaped_ghi,
            dims=("time", "y", "x"),
            coords={
                "time": ds['time'].values, 
                "y": ds['y'].values, 
                "x": ds['x'].values
            },
            name="ghi"
        )

        ghi.name = "ghi"
        ghi.attrs["units"] = "W m**-2"
        ghi.attrs["description"] = "Ghi calculated from influx diffuse (dhi) and influx direct (dni)."
        return ghi
    
    @staticmethod
    def _calculate_relative_humidity(temperature, dewpoint_temperature):
        """
        Calculates relative humidity based on air temperature and dewpoint temperature.
        Needed in order to calculate precipitable water using pvlib's :code:`gueymard94_pw()` function.

        Relative humidity is calculated using a version of the 
        August-Roche-Magnus equation as follows: 
        
        .. math::

            RH = 100 \cdot \frac{{\exp\left(\frac{{17.625 \cdot TD}}{{243.04 + TD}}\right)}}{{\exp\left(\frac{{17.625 \cdot T}}{{243.04 + T}}\right)}}

        where, RH is % relative humidity, TD is dew-point temperature (celsius), and T is air temperature (celsius).[#1]_ [#2]_

        Parameters
        ----------
        temperature : numeric
            Ambient air temperature at the surface. [C]
        dewpoint_temperature : numeric
            Dewpoint temperature at the surface. [C]

        Returns
        -------
        relative_humidity : numeric
            Percent relative humidity. [%]

        References
        ----------
        .. [#1] `United States Environmental Protection Agency. Hydrologic Micro Services. Meteorology - Humidity.  <https://qed.epa.gov/hms/meteorology/humidity/algorithms/>`_

        .. [#2] `University of Miami. Calculate Temperature, Dewpoint, or Relative Humidity. <https://bmcnoldy.earth.miami.edu/Humidity.html>` 

        """
        relative_humidity = 100 * (
            np.exp((17.625 * dewpoint_temperature) / (243.04 + dewpoint_temperature)) /
            np.exp((17.625 * temperature) / (243.04 + temperature))
        )

        relative_humidity.name = "relative_humidity"
        relative_humidity.attrs["units"] = "%"
        relative_humidity.attrs["description"] = "Relative humidity, calculated using temperature and dewpoint temperature."

        return relative_humidity
    
    @staticmethod
    def _calculate_precipitable_water(temperature, relative_humidity):
        """
        Calculates precipitable water (cm) from ambient air temperature (C) and relative humidity (%) using 
        :code:`pvlib.atmosphere.gueymard94_pw()`.  

        Precipitable water (cm) is a required input for models using CEC modules from :code:`pvlib`.
        For full documentation on how :code:`pvlib.atmosphere.gueymard94_pw()` calculates precipitable water,
        see: `the pvlib API reference for pvlib.atmosphere.gueymard94_pw() <https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.atmosphere.gueymard94_pw.html>`.

        Parameters
        ----------
        temperature : numeric
            Ambient air temperature at the surface. [C]
        relative_humidity : numeric
            Percent relative humidity. [%]

        Returns
        -------
        precipitable_water : numeric
            Precipitable water (cm) calculated from ambient air temperature (C) and relative humidity (%). [cm]

        """
        precipitable_water = gueymard94_pw(temperature, relative_humidity)
        precipitable_water.name = "precipitable_water"
        precipitable_water.attrs["units"] = "cm"
        precipitable_water.attrs["description"] = "Precipitable water (cm) calculated from ambient air temperature (C) and relative humidity (%)."

        return precipitable_water
    
    @staticmethod
    def _convert_celsius(ds):
        """
        Converts a temperature in Kelvin to a temperate in Celsius.

        Parameters
        ----------
        temperature : numeric
            A temperature in Celsius [C].

        Returns
        -------
        temperature : numeric
            A temperature in Kelvin [K].
        """
        return ds - 273.15
    
    