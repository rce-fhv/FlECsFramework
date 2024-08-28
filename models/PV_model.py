import numpy as np
import pandas as pd

from preprocessing.pv_data_download import PVData

# Old PV Model
# class PV():
#     def __init__(self, clockfun, **kwargs) -> None:
#         'This model creates a simple PV Model based on a PVGis model with hourly resolution'

#         self.clockfun = clockfun

#         defaultparams = dict(
#             latitude  = 47.440878,        # In decimal degrees, between -90 and 90, north is positive (ISO 19115)
#             longitude = 9.996436,         # In decimal degrees, between -180 and 180, east is positive (ISO 19115)
#             start = '2005',               # SEVA: time index has to start one year before the relevant year due to time zone conversion
#             end   = '2006',               # SEVA: time index goes until '2007-01-01 00:00:00+01:00'
#             surface_tilt = 0,             # Tilt angle from horizontal plane
#             surface_azimuth = 180,        # Orientation (azimuth angle) of the (fixed) plane. Clockwise from north (north=0, east=90, south=180, west=270)
#             usehorizon = True,            # Include effects of horizon
#             pvcalculation = True,         # Return estimate of hourly PV production
#             peakpower = 20,               # Nominal power of PV system in kW
#             pvtechchoice = 'crystSi',     # ({'crystSi', 'CIS', 'CdTe', 'Unknown'}, default: 'crystSi')
#             mountingplace = 'building',   # ({'free', 'building'}, default: free) – Type of mounting for PV system. Options of ‘free’ for free-standing and ‘building’ for building-integrated.
#             loss = 15,                    # (float, default: 0) – Sum of PV system losses in percent. Required if pvcalculation=True
#             optimal_surface_tilt = True,  # (bool, default: False) – Calculate the optimum tilt angle. Ignored for two-axis tracking
#             optimalangles = True,         # (bool, default: False) – Calculate the optimum tilt and azimuth angles. Ignored for two-axis tracking.
#         )
#         defaultparams.update(**kwargs)
#         self.data, inputs, self.metadata = get_pvgis_hourly(**defaultparams)
#         self.data.index = self.data.index.round('h').tz_convert('Europe/Vienna')
#         # self.data.index = self.data.index.round('h')  # .tz_localize('Europe/Vienna') # put the data to full hours (output form pvgis seems to be always at **:10:00)

#     @staticmethod
#     def download_data_to_file(pv_data_dir, **kwargs):
#         raise NotImplementedError('download_data_to_file is not yet implemented, migth give some speed improvements when running many scenarios + improved independence and reproducability')
#         # pv_data_path = pathlib.Path(pv_data_dir)
#         # data.to_pickle(pv_data_dir)

#     def step(self, time):
#         dtime = self.clockfun(time)
        # return self.data.loc[dtime, 'P']
    
PVDEFAULTDATANAME = 'pv_data'

class PV():
    def __init__(self, clockfun, eids, params_list, data_name) -> None:
        '''
        PV model based on the pvgis model via pvlib with hourly resolution.
        clockfun: clock function mapping int time steps to datetime.
        download_data: Bool: use existing data or download data.
        data_name: str. name of file to use or to create
        params_list: list of dicts with parameters for downloading pv datasets (only used when offline_data_name=None)
        '''
        pvdata = PVData()
        # if download_data:
        if pvdata.params_changed(eids, data_name, params_list):
            print('Downloading PV data from PVGIS...')
            pvdata.download_data(eids=eids, params_list=params_list, name=data_name)
        # else: 
        #     print('PV using offline data...')

        self.data = pvdata.get_offline_data(name=data_name)  # load the preprocessed data
        self.clockfun = clockfun

    def step(self, inttime):
        datetime = self.clockfun(inttime)
        row = self.data.loc[datetime]
        
        return datetime, {level: row.xs(level).to_dict() for level in row.index.levels[0]}
    

class PVperfectPrediction():
    def __init__(self, clockfun, data_name, horizon_intsteps) -> None:
        pvdata = PVData()
        self.data = pvdata.get_offline_data(name=data_name)  # load the preprocessed data
        self.data = self.data.xs('pv_power', axis=1, level=1)
        self.clockfun = clockfun
        self.horizon_intsteps = horizon_intsteps

    def step(self, inttime):
        datetime = self.clockfun(inttime)
        datetime_end = self.clockfun(inttime+self.horizon_intsteps-1)
        return self.data.loc[datetime:datetime_end, :].sum(axis=1)

    
if __name__ == '__main__':
    pass
    # clockfun = lambda int_time: pd.to_datetime(int_time * 60*60, unit='s', origin=pd.Timestamp('2006-01-01 00:00:00')).tz_localize('Europe/Vienna')
    # pv = PV(clockfun=clockfun, data_name='test_data')
    # print(pv.step(12))
