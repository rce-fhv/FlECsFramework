import pandas as pd
import numpy as np
import pathlib
import sys
from pvlib.iotools import get_pvgis_hourly
from copy import copy
sys.path.append('../../')


class PVData:
    preprocessed_data_dir = pathlib.Path('data/preprocessed')

    paramspath = preprocessed_data_dir.joinpath('pv_data_params.pkl')

    defaultparams = dict(
        latitude  = 47.440878,        # In decimal degrees, between -90 and 90, north is positive (ISO 19115)
        longitude = 9.996436,         # In decimal degrees, between -180 and 180, east is positive (ISO 19115)
        start = '2009',               # SEVA: time index has to start one year before the relevant year due to time zone conversion
        end   = '2010',               # SEVA: time index goes until '2007-01-01 00:00:00+01:00'
        surface_tilt = 0,             # Tilt angle from horizontal plane
        surface_azimuth = 180,        # Orientation (azimuth angle) of the (fixed) plane. Clockwise from north (north=0, east=90, south=180, west=270)
        usehorizon = True,            # Include effects of horizon
        pvcalculation = True,         # Return estimate of hourly PV production
        peakpower = 5,                # Nominal power of PV system in kW
        # pvtechchoice = 'crystSi',     # ({'crystSi', 'CIS', 'CdTe', 'Unknown'}, default: 'crystSi')
        # mountingplace = 'building',   # ({'free', 'building'}, default: free) – Type of mounting for PV system. Options of ‘free’ for free-standing and ‘building’ for building-integrated.
        # loss = 15,                    # (float, default: 0) – Sum of PV system losses in percent. Required if pvcalculation=True
        # optimal_surface_tilt = True,  # (bool, default: False) – Calculate the optimum tilt angle. Ignored for two-axis tracking
        # optimalangles = True,         # (bool, default: False) – Calculate the optimum tilt and azimuth angles. Ignored for two-axis tracking.
    )

    data = None
    parameters = None

    def get_offline_data(self, name):
        path =  self.preprocessed_data_dir.joinpath(name+'.pkl')
        return pd.read_pickle(path)
    
    def get_params(self, name):
        paramspath =  self.preprocessed_data_dir.joinpath(name+'_params.pkl')
        if pathlib.Path.exists(paramspath):
            return pd.read_pickle(paramspath)
        return pd.DataFrame([])
    
    def params_changed(self, eids, offline_data_name, new_params_list):
        old_params = self.get_params(offline_data_name)
        new_params = pd.DataFrame.from_dict(self.create_complete_params_dict(eids, new_params_list), orient='index')
        return not old_params.equals(new_params)
    
    def create_complete_params_dict(self, eids, params_list):
        all_params = {}
        for eid, params in zip(eids, params_list):
            # updata parameter dict and append
            defaultparams = copy(self.defaultparams)
            defaultparams.update(**params)
            all_params[eid] = defaultparams
        return all_params
    
    def download_data(self, eids, params_list, name):
        datas = {}
        all_params = self.create_complete_params_dict(eids, params_list)
        for eid, params in all_params.items():
            params = params.copy() # this is to save all parameters later (pop)
            additional_outputs = params.pop('additional_outputs', []) # enebles the download of weather data for individual solar systems

            # download data from pvgis
            data_i, inputs, metadata = get_pvgis_hourly(**params)
            # reindexing with timezone aware index rounded to full hours
            data_i.index = data_i.index.round('h').tz_convert('Europe/Vienna')
            data_i = data_i[['P']+additional_outputs].rename({'P': 'pv_power'}, axis='columns')

            datas[eid] = data_i
        self.data = pd.concat(datas, axis=1, names=['Name'])

        # save data as pickle 
        path =  self.preprocessed_data_dir.joinpath(name+'.pkl')
        self.data.to_pickle(path)

        # save parameter dict for future refernce
        self.parameters = pd.DataFrame.from_dict(all_params, orient='index')
        paramspath =  self.preprocessed_data_dir.joinpath(name+'_params.pkl')
        self.parameters.to_pickle(paramspath)


if __name__ == '__main__':
    pvdata = PVData()
    pvdata.download_data({'peakpower':5}, {'peakpower':10}, {'peakpower':15}, {'peakpower':20})