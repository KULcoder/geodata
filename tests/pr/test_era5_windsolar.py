# Copyright 2025 Keyu Long (UCSD)

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
from dask.distributed import Client

import xarray as xr

from geodata.datasets import load_dataset
from geodata.logging import logger
from geodata.model.pvlib import PVLib

# Set logger to DEBUG level to see all debug messages
logger.setLevel(logging.DEBUG)

def test_wind_solar_workflow():
    """Test that the wind interpolation workflow completes without errors.
    
    This test verifies:
    - Dataset can be loaded and downloaded
    - Model can be created and prepared
    - Capacity factor estimation works (globally and with bounds)
    - Wind speed estimation works at a specific height
    - Results can be computed and have valid values
    """

    client = Client(processes=True, threads_per_worker=1)

    years = slice(2016, 2016)
    months = slice(1, 2)

    ds_cls = load_dataset("wind_solar_hourly")
    ds = ds_cls(years=years, months=months, testing=True)

    ds.download()
    assert ds.downloaded, "Dataset should be downloaded successfully"

    # Create model with the dataset
    model = PVLib(ds)
    assert model is not None, "Model should be created successfully"

    china_bbox = (73.5, 18.2, 135.1, 53.6)  # China bounding box
    xs = slice(china_bbox[0], china_bbox[2])
    ys = slice(china_bbox[3], china_bbox[1])
    years = slice(2016, 2016)
    months = slice(1, 1)

    # TODO: add a test here to test that
    # the model must not estimate without pv_system and model_config


    # create the pv_system
    n_mods = 50
    n_strings = 1
    cec_modules = model.retrieve_sam('CECMod')
    module = cec_modules['Kaneka_U_SA105']
    inv = model.retrieve_sam("CECInverter")['Fronius_USA__CL_33_3_Delta__208V_']
    model.init_pv_system(
        arrays = None,
        surface_tilt=35,
        surface_azimuth=180,
        racking_model = 'open_rack',
        module_parameters=module,
        modules_per_string = n_mods,
        module_type = 'glass_polymer',
        module = 'Kaneka_U_SA105',
        strings_per_inverter = n_strings, 
        inverter_parameters=inv
    )

    assert model.pv_system is not None, "pv_system should be seccesfully created"
    # TODO: assert it to the correct type

    # create the model_config
    model.init_model_config(
        clearsky_model= 'haurwitz',
        transposition_model='perez', 
        solar_position_method= 'nrel_numpy',
        airmass_model= 'kastenyoung1989',
        dc_model='cec',
        ac_model='sandia', 
        aoi_model="physical",
        spectral_model='first_solar',
        dc_ohmic_model='no_loss'
    )
    assert model.config is not None, "model config should be successfully created"
    # TODO: assert it to the correct type


    # Test capacity factor estimation globally
    ac_power_and_pv_capacity_global = model.estimate(years=years, months=months)
    assert ac_power_and_pv_capacity_global is not None, "Capacity factor estimation should return a result"
    assert isinstance(ac_power_and_pv_capacity_global, (xr.DataArray, xr.Dataset)), \
        "Capacity factor should be an xarray DataArray or Dataset"
    
    # Test capacity factor estimation for China only
    ac_power_and_pv_capacity_china = model.estimate(years=years, months=months, xs=xs, ys=ys)
    assert ac_power_and_pv_capacity_china is not None, "Capacity factor estimation with bounds should return a result"
    assert isinstance(ac_power_and_pv_capacity_china, (xr.DataArray, xr.Dataset)), \
        "Capacity factor with bounds should be an xarray DataArray or Dataset"

    # Test that results can be computed
    output_computed = ac_power_and_pv_capacity_china.compute()
    assert output_computed is not None, "Computed capacity factor should not be None"
    
    # TODO: design correct output test specific regard to the pvlib output

    client.close()