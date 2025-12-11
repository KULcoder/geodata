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
import pprint
import shutil
import tempfile
import zipfile
from pathlib import Path

import xarray as xr

from ..._base import AtomicDataset
from ._base import ERA5WindSolarBaseDataset

logger = logging.getLogger(__name__)


class ERA5WindSolarHourlyDataset(ERA5WindSolarBaseDataset):
    """ERA5WindSolarHourlyDataset is a class that handles the downloading,
    preprocessing, and storing of the ERA5 dataset for wind and solar
    information. This dataset is stored in hourly intervals.

    The ERA5 dataset is a reanalysis dataset that provides a comprehensive
    record of the Earth's climate. It is produced by the European Centre for
    Medium-Range Weather Forecasts (ECMWF) and is available from 1980 to
    present.

    Note:
        - The specific variables that are downloaded are:
            - 100m_u_component_of_wind
            - 100m_v_component_of_wind
            - 2m_temperature
            - 2m_dewpoint_temperature
            - runoff
            - soil_temperature_level_4
            - surface_net_solar_radiation
            - surface_pressure
            - surface_solar_radiation_downwards
            - toa_incident_solar_radiation
            - total_sky_direct_solar_radiation_at_surface
            - forecast_surface_roughness
            - geopotential
    """

    weather_config = "wind_solar_hourly"

    # Information that is needed for ERA5's API request
    variables = {
        "100m_u_component_of_wind": "u100",
        "100m_v_component_of_wind": "v100",
        "2m_temperature": "t2m",
        "2m_dewpoint_temperature": "d2m",
        "runoff": "ro",
        "soil_temperature_level_4": "stl4",
        "surface_net_solar_radiation": "ssr",
        "surface_pressure": "sp",
        "surface_solar_radiation_downwards": "ssrd",
        "toa_incident_solar_radiation": "tisr",
        "total_sky_direct_solar_radiation_at_surface": "fdir",
        "forecast_surface_roughness": "fsr",
        "geopotential": "z",
    }
    product = "reanalysis-era5-single-levels"
    product_type = "reanalysis"

    def _save_extracted_files_for_debugging(self, tempdir: str, zip_path: str, save_path: Path):
        """Save extracted NetCDF files and zip file to a debug directory for inspection.
        
        Args:
            tempdir: Temporary directory containing extracted files
            zip_path: Path to the downloaded zip file
            save_path: Path where the final file will be saved
            
        Returns:
            List of saved file paths
        """
        debug_dir = save_path.parent / "debug_extracted_files"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the zip file
        zip_dest = debug_dir / "download.zip"
        shutil.copy2(zip_path, zip_dest)
        logger.info(f"Saved zip file: {zip_dest} (size: {os.path.getsize(zip_dest)} bytes)")
        
        # Save extracted NetCDF files
        nc_files = [
            os.path.join(tempdir, f)
            for f in os.listdir(tempdir)
            if f.endswith(".nc")
        ]
        
        saved_files = [str(zip_dest)]
        logger.info(f"Saving {len(nc_files)} extracted files to {debug_dir} for debugging")
        for nc_file in nc_files:
            dest_file = debug_dir / os.path.basename(nc_file)
            shutil.copy2(nc_file, dest_file)
            saved_files.append(str(dest_file))
            logger.info(f"  Saved: {dest_file} (size: {os.path.getsize(dest_file)} bytes)")
        
        return saved_files

    @staticmethod
    def test_netcdf_files(file_paths):
        """Test if NetCDF files are corrupted or can be opened.
        
        Args:
            file_paths: List of file paths to test (can be str or Path)
            
        Returns:
            dict: Results with 'valid_files', 'corrupted_files', and 'errors'
        """
        results = {
            'valid_files': [],
            'corrupted_files': [],
            'errors': {}
        }
        
        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                results['errors'][str(file_path)] = "File does not exist"
                results['corrupted_files'].append(str(file_path))
                continue
            
            file_size = file_path.stat().st_size
            if file_size == 0:
                results['errors'][str(file_path)] = "File is empty (0 bytes)"
                results['corrupted_files'].append(str(file_path))
                continue
            
            # Try to open the file with h5netcdf engine (required for ERA5 files)
            try:
                with xr.open_dataset(file_path, decode_times=False, engine='h5netcdf') as ds:
                    # Try to access basic properties
                    dims = ds.dims
                    coords = list(ds.coords.keys())
                    data_vars = list(ds.data_vars.keys())
                    
                results['valid_files'].append({
                    'path': str(file_path),
                    'size': file_size,
                    'dims': dict(dims),
                    'coords': coords,
                    'data_vars': data_vars
                })
                logger.info(f"✓ {file_path.name}: Valid ({file_size} bytes, {len(data_vars)} variables)")
                
            except Exception as e:
                error_msg = str(e)
                results['errors'][str(file_path)] = error_msg
                results['corrupted_files'].append(str(file_path))
                logger.error(f"✗ {file_path.name}: Corrupted - {error_msg}")
        
        return results

    def _download_file(self, file: AtomicDataset):
        year: int = file.year
        month: int = file.month
        save_path: Path = file.path

        # For testing/debugging: limit to first 3 days and first 3 hours
        if self.testing:
            days = [f"{d:02d}" for d in range(1, 4)]  # Days 1-3
            times = [f"{t:02d}:00" for t in range(0, 3)]  # Hours 0-2
            logger.info(
                f"Testing mode: Downloading only days {days} and times {times} "
                f"for faster debugging"
            )
        else:
            days = [f"{d:02d}" for d in range(1, 32)]  # All days in month
            times = [f"{t:02d}:00" for t in range(0, 24)]  # All hours

        full_request = {
            "product_type": self.product_type,
            "format": "netcdf",
            "variable": list(self.variables.keys()),
            "year": year,
            "month": month,
            "day": days,
            "time": times,
        }

        if self.bounds is not None:
            full_request["area"] = self.bounds[::-1]

        logger.debug("Full request for download: %s", pprint.pformat(full_request))

        full_result = self.client.retrieve(self.product, full_request)
        if full_result.content_type == "application/zip":
            logger.info(
                "Multiple files found with request. Additional unzipping/preprocessing needed."
            )

            with tempfile.TemporaryDirectory() as tempdir:
                zip_path = os.path.join(tempdir, "download.zip")
                full_result.download(zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tempdir)

                nc_files = [
                    os.path.join(tempdir, f)
                    for f in os.listdir(tempdir)
                    if f.endswith(".nc")
                ]
                
                # Save extracted files for debugging if in testing mode
                if self.testing:
                    saved_files = self._save_extracted_files_for_debugging(tempdir, zip_path, save_path)
                    logger.info(f"Extracted files saved to debug directory. You can test them with:")
                    logger.info(f"  from geodata.datasets.era5.wind_solar.hourly import ERA5WindSolarHourlyDataset")
                    logger.info(f"  results = ERA5WindSolarHourlyDataset.test_netcdf_files({saved_files[1:]})  # Skip zip file")
                    logger.info(f"  print(results)")

                # Open with h5netcdf engine (required for ERA5 files)
                # Write with h5netcdf engine as well to avoid HDF compatibility issues
                # This allows writing without loading entire dataset into memory
                with xr.open_mfdataset(nc_files, engine='h5netcdf') as ds:
                    ds.to_netcdf(save_path, engine='h5netcdf')

                logger.info("Preprocessing complete with zipfile")
                logger.info("Successfully downloaded to %s", save_path)

