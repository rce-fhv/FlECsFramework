import numpy as np
import pandas as pd
from preprocessing.german_household_data_preprocessing import LoadData

LOADDEFAULTDATANAME = 'load_data'

class LoadModel():
    def __init__(self, clockfun, data_name=LOADDEFAULTDATANAME, num=0) -> None:
        '''
        Load model for simpe household load simulation based on csv load profiles. Load power is negative
        clockfun: clock function mapping int time steps to datetime.
        data_name: str. name of file to use or to create
        num: number of load profiles to use, 
        '''
        # TODO: provide option, to select specific columns from the original data when preprocesing
        loaddata = LoadData()
        data = loaddata.get_preprocessed_data(name=data_name)  # load the preprocessed data (returns an empty df when it does not exist)
        
        # if len(data.columns) != num:
        loaddata.preprocess_data(num, data_name)
        data = loaddata.get_preprocessed_data(name=data_name)

        self.data = data
        self.clockfun = clockfun

    def step(self, inttime):
        datetime = self.clockfun(inttime)
        return datetime, dict(self.data.loc[datetime, :])
    

class LoadModelPerfectPredictor():
    def __init__(self, clockfun, horizon_intsteps, data_name=LOADDEFAULTDATANAME) -> None:
        # print('getting load data for prediction...')
        loaddata = LoadData()  # load the preprocessed data
        self.data = loaddata.get_preprocessed_data(data_name) # Load data is negative Power
        self.clockfun = clockfun
        self.horizon_intsteps = horizon_intsteps

    def step(self, inttime):
        datetime_curent_time = self.clockfun(inttime)
        datetime_horizon_end = self.clockfun(inttime+self.horizon_intsteps-1)
        return self.data.loc[datetime_curent_time:datetime_horizon_end, :].sum(axis=1)
