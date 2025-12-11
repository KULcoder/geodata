# Copyright 2024-2025 Michael Davidson (UCSD), Xiqiang Liu (UCSD), Keyu Long (UCSD)

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

import logging
import os
from pathlib import Path

import xarray as xr
import numpy as np

from geodata.types import PathLike
from .._base import ERA5BaseDataset, _subset_x_y_era5

logger = logging.getLogger(__name__)

def _add_height(ds):
    """Convert geopotential 'z' to geopotential height following [1]

    References
    ----------
    [1] ERA5: surface elevation and orography, retrieved: 10.02.2019
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+surface+elevation+and+orography

    """
    g0 = 9.80665
    z = ds["z"]
    if "time" in z.coords:
        z = z.isel(time=0, drop=True)
    ds["height"] = z / g0
    ds = ds.drop("z")
    return ds


def preprocess_wind_solar_dataset(ds: xr.Dataset, compute_binary_ops: bool = False) -> xr.Dataset:
    """Preprocess ERA5 wind-solar dataset to convert raw variables to processed format.
    
    This function applies the same preprocessing logic used in prepare_func, but works
    on an already-loaded dataset. This allows models to reuse the preprocessing logic
    without duplicating code.
    
    Args:
        ds: Raw ERA5 wind-solar dataset with variables like u100, v100, t2m, fdir, etc.
        compute_binary_ops: If True, compute variables needed for binary operations
            to avoid file handle issues with lazy arrays and parallel reading.
    
    Returns:
        Preprocessed dataset with variables like influx_diffuse, influx_direct, 
        wnd100m, temperature, etc.
    """
    # Step 1: Convert geopotential 'z' to geopotential height 'height'
    if 'z' in ds.data_vars:
        ds = _add_height(ds)
    
    # Step 2: Rename variables
    if 'fdir' in ds.data_vars:
        ds = ds.rename({"fdir": "influx_direct"})
    if 'tisr' in ds.data_vars:
        ds = ds.rename({"tisr": "influx_toa"})
    
    # Compute variables needed for binary operations if requested
    # This avoids file handle issues with parallel reading when doing coordinate merging.
    # We load these variables right before they're used in binary operations to ensure
    # both operands are in memory, avoiding file access during coordinate merging.
    if compute_binary_ops:
        vars_to_load = set()
        if 'ssrd' in ds.data_vars and 'ssr' in ds.data_vars:
            vars_to_load.update(['ssrd', 'ssr'])
        if 'ssrd' in ds.data_vars and 'influx_direct' in ds.data_vars:
            vars_to_load.update(['ssrd', 'influx_direct'])
        if 'u100' in ds.data_vars and 'v100' in ds.data_vars:
            vars_to_load.update(['u100', 'v100'])
        
        # Load variables (removes duplicates automatically via set)
        for var in vars_to_load:
            if var in ds.data_vars:
                ds[var] = ds[var].load()
    
    # Step 3: Calculate albedo
    if 'ssrd' in ds.data_vars and 'ssr' in ds.data_vars:
        with np.errstate(divide="ignore", invalid="ignore"):
            ds["albedo"] = (
                ((ds["ssrd"] - ds["ssr"]) / ds["ssrd"])
                .fillna(0.0)
                .assign_attrs(units="(0 - 1)", long_name="Albedo")
            )
    
    # Step 4: Calculate influx_diffuse from ssrd and influx_direct
    if 'ssrd' in ds.data_vars and 'influx_direct' in ds.data_vars:
        ds["influx_diffuse"] = (ds["ssrd"] - ds["influx_direct"]).assign_attrs(
            units="J m**-2", long_name="Surface diffuse solar radiation downwards"
        )
    
    # Step 5: Drop ssrd and ssr (after using them)
    if 'ssrd' in ds.data_vars:
        ds = ds.drop("ssrd")
    if 'ssr' in ds.data_vars:
        ds = ds.drop("ssr")
    
    # Step 6: Convert from energy to power J m**-2 -> W m**-2 and clip negative fluxes
    for a in ("influx_direct", "influx_diffuse", "influx_toa"):
        if a in ds.data_vars:
            ds[a] = ds[a].clip(min=0.0) / (60.0 * 60.0)
            ds[a].attrs["units"] = "W m**-2"
    
    # Step 7: Calculate wnd100m from u100 and v100
    if 'u100' in ds.data_vars and 'v100' in ds.data_vars:
        # Use xarray operations to ensure we get a DataArray with attrs
        wnd100m = (ds["u100"] ** 2 + ds["v100"] ** 2) ** 0.5
        wnd100m.attrs.update({
            "units": ds["u100"].attrs.get("units", "m s**-1"),
            "long_name": "100 metre wind speed"
        })
        ds["wnd100m"] = wnd100m
        ds = ds.drop(["u100", "v100"])
    
    # Step 8: Rename remaining variables
    rename_map = {}
    if 'ro' in ds.data_vars:
        rename_map["ro"] = "runoff"
    if 't2m' in ds.data_vars:
        rename_map["t2m"] = "temperature"
    if 'sp' in ds.data_vars:
        rename_map["sp"] = "pressure"
    if 'stl4' in ds.data_vars:
        rename_map["stl4"] = "soil temperature"
    if 'fsr' in ds.data_vars:
        rename_map["fsr"] = "roughness"
    
    if rename_map:
        ds = ds.rename(rename_map)
    
    # Step 9: Handle valid_time -> time rename (for new ERA5 format)
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    
    return ds

class ERA5WindSolarBaseDataset(ERA5BaseDataset):
    """Base class for ERA5 wind and solar datasets.
    
    This class provides the prepare_func implementation specific to wind_solar datasets,
    which use single-level data from the reanalysis-era5-single-levels product.
    """

    @classmethod
    def is_preprocessed(cls, ds: xr.Dataset) -> bool:
        """Check if a dataset is already preprocessed.
        
        A dataset is considered preprocessed if it contains processed variables
        (like influx_diffuse, influx_direct, wnd100m, temperature) and does not
        contain raw variables (like u100, v100, t2m, fdir, ssrd, ssr, z).
        
        Args:
            ds: Dataset to check
            
        Returns:
            True if dataset is preprocessed, False otherwise
        """
        # Check for raw variables that indicate preprocessing is needed
        has_raw_vars = any(var in ds.data_vars for var in ['u100', 'v100', 't2m', 'fdir', 'ssrd', 'ssr', 'z'])
        
        # Check for processed variables that indicate preprocessing is done
        has_processed_vars = any(var in ds.data_vars for var in [
            'influx_diffuse', 'influx_direct', 'wnd100m', 'temperature', 'height'
        ])
        
        # If we have processed vars and no raw vars, it's preprocessed
        if has_processed_vars and not has_raw_vars:
            return True
        
        # If we have raw vars, it's not preprocessed
        if has_raw_vars:
            return False
        
        # If we have neither, check if time coordinate is renamed (indicates preprocessing)
        if "time" in ds.coords and "valid_time" not in ds.coords:
            # Might be preprocessed, but we can't be sure without processed vars
            # Return False to be safe
            return False
        
        # Default: assume not preprocessed if we can't determine
        return False

    @classmethod
    def ensure_preprocessed(
        cls,
        ds: xr.Dataset,
        save_path: PathLike | None = None,
        force: bool = False,
    ) -> xr.Dataset:
        """Ensure a dataset is preprocessed, preprocessing if necessary.
        
        This method checks if the dataset is preprocessed, and if not, applies
        preprocessing. Optionally saves the preprocessed dataset to disk.
        
        Args:
            ds: Dataset to check and preprocess if needed
            save_path: Optional path to save preprocessed dataset. If None, dataset
                is not saved to disk.
            force: If True, force preprocessing even if dataset appears preprocessed
            
        Returns:
            Preprocessed dataset
        """
        # Check if preprocessing is needed
        if not force and cls.is_preprocessed(ds):
            logger.debug("Dataset is already preprocessed, skipping preprocessing.")
            return ds
        
        logger.info("Dataset needs preprocessing, applying preprocessing...")
        
        # Binary operations on lazy arrays (like subtraction) require coordinate merging,
        # which triggers file access and can cause HDF5/netCDF4 file handle issues with parallel reading.
        # To avoid loading the entire dataset (which causes memory exhaustion),
        # we pass a flag to preprocess_wind_solar_dataset to compute only the variables
        # needed for binary operations, right before those operations are performed.
        # Other operations (renames, drops) remain lazy.
        logger.debug("Applying preprocessing with selective computation of variables for binary operations...")
        ds_preprocessed = preprocess_wind_solar_dataset(ds, compute_binary_ops=True)
        
        # Optionally save to disk
        if save_path is not None:
            logger.info(f"Saving preprocessed dataset to {save_path}")
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to temporary file first, then rename (xarray doesn't support overwriting)
            temp_path = save_path.with_stem(save_path.stem + "_preprocessed")
            ds_preprocessed.to_netcdf(temp_path)
            
            # Replace original file if it exists
            if save_path.exists():
                save_path.unlink()
            temp_path.rename(save_path)
            
            logger.info(f"Preprocessed dataset saved to {save_path}")
        
        return ds_preprocessed

    @classmethod
    def prepare_func(
        cls,
        fn: PathLike,
        year: int,
        month: int,
        xs: slice,
        ys: slice,
        **kwargs,
    ):
        """Prepare the dataset for a given year and month.
        
        This implementation is specific to wind_solar datasets which:
        - Use single-level data (no model levels)
        - Download from reanalysis-era5-single-levels product
        - Are stored as monthly files
        """
        if isinstance(fn, str) and not os.path.exists(fn):
            return
        if isinstance(fn, list) and not all(os.path.isfile(f) for f in fn):
            return

        with xr.open_dataset(fn) as ds:
            logger.info("Opening %s", fn)
            ds = _subset_x_y_era5(ds, xs, ys)
            
            # Apply preprocessing using shared function
            ds = preprocess_wind_solar_dataset(ds)

            yield (year, month), ds

