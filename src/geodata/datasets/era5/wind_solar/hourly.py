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

    def _download_file(self, file: AtomicDataset):
        year: int = file.year
        month: int = file.month
        save_path: Path = file.path

        full_request = {
            "product_type": self.product_type,
            "format": "netcdf",
            "variable": list(self.variables.keys()),
            "year": year,
            "month": month,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{t:02d}:00" for t in range(0, 24)],
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
                _count = 0
                for _ in range(3):
                    try:
                        _count += 1
                        full_result.download(os.path.join(tempdir, "download.zip"))
                        break
                    except Exception as e:
                        logger.error("Error downloading file: %s", e)
                        if _count == 3:
                            raise

                with zipfile.ZipFile(
                    os.path.join(tempdir, "download.zip"), "r"
                ) as zip_ref:
                    zip_ref.extractall(tempdir)

                # Check each extracted NetCDF file for corruption before opening
                nc_files = [
                    os.path.join(tempdir, f)
                    for f in os.listdir(tempdir)
                    if f.endswith(".nc")
                ]
                
                valid_files = []
                corrupted_files = []
                
                for nc_file in nc_files:
                    try:
                        # Try to open the file to check if it's valid
                        with xr.open_dataset(nc_file) as test_ds:
                            # Just verify it can be opened, don't load data
                            _ = test_ds.dims
                        valid_files.append(nc_file)
                    except (OSError, IOError) as e:
                        logger.warning(
                            f"Corrupted NetCDF file detected in zip: {nc_file}. "
                            f"Error: {e}. Skipping this file."
                        )
                        corrupted_files.append(nc_file)
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error checking file {nc_file}: {e}. "
                            "Skipping this file."
                        )
                        corrupted_files.append(nc_file)
                
                if not valid_files:
                    error_msg = (
                        f"All {len(nc_files)} extracted NetCDF files are corrupted. "
                        "Cannot proceed with download."
                    )
                    logger.error(error_msg)
                    raise OSError(error_msg)
                
                if corrupted_files:
                    logger.warning(
                        f"Skipping {len(corrupted_files)} corrupted file(s) out of "
                        f"{len(nc_files)} total. Proceeding with {len(valid_files)} valid file(s)."
                    )

                with xr.open_mfdataset(valid_files) as ds:
                    ds.to_netcdf(save_path)

                logger.info("Preprocessing complete with zipfile")
                logger.info("Successfully downloaded to %s", save_path)

