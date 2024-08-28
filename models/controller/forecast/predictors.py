import numpy as np
import pandas as pd

from models.PV_model import PVperfectPrediction, PVDEFAULTDATANAME
from models.load_model import LoadModelPerfectPredictor, LOADDEFAULTDATANAME
from models.controller.forecast.LoadModelLstm import LoadModelLstm

class ResidualPowerPrediction:
    def __init__(self, horizon_intsteps=24) -> None:
        raise NotImplementedError('PowerPrediction not implemented')

    def update_prediction_model(self, historic_residual_load, **kwargs):  # historic_production, historic_weather
        '''
        Updates the prediction model with the latest historic residual load data
        dependent on the type of prediciton model, this can either trigger a training process of the model or just store the data.
        '''
        raise NotImplementedError('update_prediction_model not implemented')

    def update_prediction(self, time, **kwargs):
        '''
        Update the prediction
        Create a prediction for the time horizon specified in horizon_intsteps
        '''
        raise NotImplementedError('update_prediction not implemented')

    def get_current_predictionError(self, datetime) -> pd.Series:
        '''
        Return the current prediction as numpy array. Index 0 of the array is datetime (as provided).
        The length is the complete prediction horizon as provided by update_prediction.
        (in case update_predicion fails, the time horizon can be shorter)
        '''
        raise NotImplementedError('get_current_prediction not implemented')


class PerfectResidualPowerPrediction(ResidualPowerPrediction):
    def __init__(self, clockfun, pv_data_name=PVDEFAULTDATANAME, load_data_name=LOADDEFAULTDATANAME, horizon_intsteps=24) -> None:
        # create one prefect prediction instance for PV and Load
        self.clockfun = clockfun
        self.horizon_intsteps = horizon_intsteps
        self.pv_perfect_predictor = PVperfectPrediction(self.clockfun, data_name=pv_data_name, horizon_intsteps=horizon_intsteps)
        self.load_perfect_predictor = LoadModelPerfectPredictor(self.clockfun, data_name=load_data_name, horizon_intsteps=horizon_intsteps)

    def update_prediction_model(self, historic_residual_load, **_):
        pass

    def update_prediction(self, time, **_):
        pv_pred = self.pv_perfect_predictor.step(time)
        load_pred = self.load_perfect_predictor.step(time)
        self.residual_prediction = pv_pred + load_pred

    def get_current_prediction(self, time) -> pd.Series:
        datetime = self.clockfun(time)
        datetime_end = self.clockfun(time+self.horizon_intsteps-1)
        if not datetime in self.residual_prediction.index:
            raise ValueError('Time not in prediction. Most likely the prediction was not updated correctly')
        return self.residual_prediction[datetime:datetime_end]
    

class PersistenceResidualPowerPrediction(ResidualPowerPrediction):
    def __init__(self, clockfun, horizon_intsteps=24, shift_periods=48) -> None:
        # create one prefect prediction instance for PV and Load
        self.clockfun = clockfun
        self.horizon_intsteps = horizon_intsteps
        self.shift_periods = shift_periods
        self.prediction = pd.DataFrame([])

    def update_prediction_model(self, historic_residual_load:pd.Series, **_):
        historic_residual_load = historic_residual_load.shift(periods=self.shift_periods, freq='infer')
        self.prediction = pd.concat([self.prediction, historic_residual_load[~historic_residual_load.index.isin(self.prediction.index)]], axis='index', verify_integrity=True)  # , join='inner'

    def update_prediction(self, time, **_):
        pass

    def get_current_prediction(self, time) -> pd.Series:
        datetime = self.clockfun(time)
        datetime_end = self.clockfun(time+self.horizon_intsteps-1)
        if not datetime in self.prediction.index:
            index = pd.date_range(datetime, datetime_end, self.horizon_intsteps)
            return pd.Series(np.zeros(self.horizon_intsteps), index=index) # when no data available (e.g. in the beinning, return zeros)
            # raise ValueError('Time not in prediction. Most likely the prediction was not updated correctly')
        return self.prediction.loc[datetime:datetime_end, 0]


class PersistenceResidualPowerPrediction1(ResidualPowerPrediction):
    def __init__(self, clockfun, horizon_intsteps=24, shift_periods_load=24*7, shift_periods_pv=24) -> None:
        # create one prefect prediction instance for PV and Load
        self.clockfun = clockfun
        self.horizon_intsteps = horizon_intsteps
        self.shift_periods_load = shift_periods_load
        self.shift_periods_pv = shift_periods_pv
        self.prediction = pd.DataFrame([])

    def update_prediction_model(self, historic_residual_load, historic_power_generation, historic_weather_data, **_):
        historic_load = historic_residual_load - historic_power_generation
        self.load_prediction = historic_load.shift(periods=self.shift_periods_load, freq='infer')
        
    def update_prediction(self, time, historic_power_generation, **_):
        self.pv_prediction = historic_power_generation.shift(periods=self.shift_periods_pv, freq='infer')
        self.prediction = (self.load_prediction + self.pv_prediction).dropna()

    def get_current_prediction(self, time) -> pd.Series:
        datetime = self.clockfun(time)
        datetime_end = self.clockfun(time+self.horizon_intsteps-1)
        if not datetime in self.prediction.index:
            index = pd.date_range(datetime, datetime_end, self.horizon_intsteps)
            return pd.Series(np.zeros(self.horizon_intsteps), index=index) # when no data available (e.g. in the beinning, return zeros)
            # raise ValueError('Time not in prediction. Most likely the prediction was not updated correctly')
        return self.prediction.loc[datetime:datetime_end]
    

class LoadForecastWithPerfectPvPrediction(ResidualPowerPrediction):
    def __init__(self, clockfun, pv_data_name=PVDEFAULTDATANAME, load_data_name=LOADDEFAULTDATANAME, horizon_intsteps=24, limit_history=None) -> None:
        self.clockfun = clockfun
        self.horizon_intsteps = horizon_intsteps
        self.limit_history = limit_history
        self.pv_perfect_predictor = PVperfectPrediction(self.clockfun, data_name=pv_data_name, horizon_intsteps=horizon_intsteps)
        self.load_predictor = LoadModelLstm(self.clockfun, data_name=load_data_name, horizon_intsteps=horizon_intsteps)
        self.prediction_model_is_trained = False
        self.train_weekly = True

    def update_prediction_model(self, historic_residual_load, historic_power_generation, historic_weather_data, **_):
        if self.limit_history is not None:
            now = historic_residual_load.index[-1]
            data_starttime = now-self.limit_history  # pd.Timedelta(days=60)
        historic_load = (historic_residual_load - historic_power_generation).dropna()
        self.load_predictor.train(historic_load[data_starttime:], historic_weather_data[data_starttime:])
        self.prediction_model_is_trained = True

    def update_prediction(self, time, historic_weather_data, historic_residual_load, historic_power_generation, **_):
        datetime = self.clockfun(time)
        datetime_end = self.clockfun(time+self.horizon_intsteps-1)
        index = pd.date_range(datetime, datetime_end, self.horizon_intsteps)

        if self.prediction_model_is_trained: # if the model is not trained at least once, set zeroes for the prediction horizon
            if self.limit_history is not None:
                now = historic_residual_load.index[-1]
                data_starttime = now-self.limit_history  # pd.Timedelta(days=60)
            historic_load = (historic_residual_load - historic_power_generation).dropna()
            pv_pred = self.pv_perfect_predictor.step(time)
            load_pred_nd = self.load_predictor.predict(time, historic_weather_data[data_starttime:], historic_load_data=historic_load[data_starttime:])
            load_pred  = pd.Series(load_pred_nd,  index=index)
            self.residual_prediction = pv_pred + load_pred
        else:
            self.residual_prediction =  pd.Series(np.zeros(self.horizon_intsteps), index=index)

    def get_current_prediction(self, time) -> pd.Series:
        datetime = self.clockfun(time)
        if not datetime in self.residual_prediction.index:
            raise ValueError('Time not in prediction. Most likely the prediction was not updated correctly')
        return self.residual_prediction[datetime:]
