# import tensorflow.keras as keras
# import unittest
# import numpy as np
# import pandas as pd
# from models.controller.forecast import LoadModelLstm
# import models.weather_data as weather_data
# # from models.load_model import LOADDEFAULTDATANAME

# class TestLstmIntegration(unittest.TestCase):
    
#     def test_smoketest(self):
        
#         now = pd.to_datetime('2010-07-07 09:00').tz_localize('UTC').tz_convert('Europe/Vienna')
        
#         def myClock(int_time):
#             return now
    
#         self.load_predictor = LoadModelLstm.LoadModelLstm(myClock, data_name=None, horizon_intsteps=24)
        
#         # Readout the preprocessed power profiles
#         #
#         household_loads_train = pd.read_pickle('data/preprocessed/loadprofiles_1h_train.pkl')
#         household_loads_train['Aggregated_Power'] = household_loads_train.sum(axis=1)
#         powerProfile = -1.0 * household_loads_train['Aggregated_Power']
#         powerProfile = powerProfile.loc[:now]

#         # Readout the weather data
#         #
#         startDate = powerProfile.index[0].to_pydatetime().replace(tzinfo=None)
#         endDate = powerProfile.index[-1].to_pydatetime().replace(tzinfo=None)
#         weather_measurements = weather_data.WeatherMeasurements()
#         weatherData = weather_measurements.get_data(
#                     startDate = startDate, 
#                     endDate = endDate,
#                     lat = 51.4817,      # Location:
#                     lon = 7.2165,       # Bochum Germany,
#                     alt = 102,          # Meteostat weatherstation   
#                     sample_periode = 'hourly', 
#                     tz = 'Europe/Vienna',
#                     )
#         selected_columns = ['temp', 'prcp', 'wspd', 'tsun']     # select weather data
#         weatherData = weatherData.loc[:now, selected_columns]

#         # Train and predict
#         self.load_predictor.train(powerProfile, weatherData)
#         load_pred = self.load_predictor.predict(0, weatherData)
#         self.assertTrue(load_pred is not None)
    
#     def test_prediction_mse(self):
        
#         predictions = [
#             {'time': '2010-01-01 00:00', 'expected_mse' : 1.0},   # High expected loss, because of missing pretraining
#             {'time': '2010-03-07 18:00', 'expected_mse' : 0.4}, 
#             {'time': '2010-09-01 11:00', 'expected_mse' : 0.4}, 
#             {'time': '2010-05-07 14:00', 'expected_mse' : 0.4}, 
#             {'time': '2010-12-30 23:00', 'expected_mse' : 0.4},
#             {'time': '2010-11-30 20:00', 'expected_mse' : 0.4}
#             ]
        
#         for prediction in predictions:
            
#             print("--- Test prediction at timestep", prediction['time'])  
                  
#             now = pd.to_datetime(prediction['time']).tz_localize('UTC').tz_convert('Europe/Vienna')
        
#             def myClock(int_time):
#                 return now
        
#             self.load_predictor = LoadModelLstm.LoadModelLstm(myClock, data_name=None, horizon_intsteps=24)
            
#             # Readout the preprocessed power profiles
#             #
#             household_loads_train = pd.read_pickle('data/preprocessed/loadprofiles_1h_train.pkl')
#             household_loads_train['Aggregated_Power'] = household_loads_train.sum(axis=1)
#             powerProfile_all = -1.0 * household_loads_train['Aggregated_Power']
            
#             # Get the power of the last 2 month. And unroll it.
#             roll_over_data = powerProfile_all.loc['2010-11':'2010-12'].copy()
#             roll_over_data.index = roll_over_data.index - pd.DateOffset(years=1)
#             powerProfile_all = pd.concat([roll_over_data, powerProfile_all])
#             powerProfile = powerProfile_all.loc[now-pd.Timedelta(days=61):now]

#             # Readout the weather data
#             #
#             startDate = powerProfile.index[0].to_pydatetime().replace(tzinfo=None)
#             endDate = powerProfile.index[-1].to_pydatetime().replace(tzinfo=None)
#             weather_measurements = weather_data.WeatherMeasurements()
#             weatherData = weather_measurements.get_data(
#                         startDate = startDate, 
#                         endDate = endDate,
#                         lat = 51.4817,      # Location:
#                         lon = 7.2165,       # Bochum Germany,
#                         alt = 102,          # Meteostat weatherstation   
#                         sample_periode = 'hourly', 
#                         tz = 'Europe/Vienna',
#                         )
#             selected_columns = ['temp', 'prcp', 'wspd', 'tsun']     # select weather data
#             weatherData = weatherData.loc[:now, selected_columns]

#             # Train and predict
#             self.load_predictor.train(powerProfile, weatherData)
#             load_pred = self.load_predictor.predict(0, weatherData)
            
#             # Evaluate the prediction
#             self.assertTrue(load_pred is not None)
#             load_real = np.array(powerProfile_all.loc[now:now+pd.Timedelta(days=0, hours=23)].values)
#             self.assertTrue(load_real.shape == (24,), "load_real.shape == " + str(load_real.shape))
#             prediction_mse = np.mean((self.load_predictor.lstmAdapter.normalizeY(load_pred) 
#                                     - self.load_predictor.lstmAdapter.normalizeY(load_real))**2)
#             print("Prediction MSE", prediction_mse)
#             self.assertTrue(prediction_mse <= prediction['expected_mse'], "Prediction MSE unexpected high: " + str(prediction_mse))
        
# if __name__ == '__main__':
#     unittest.main()
