import numpy as np
import pandas as pd
import holidays
import pytz
import pathlib

# Imports own modules.
# All imports are done relative to the root of the project.
import models.controller.forecast.Model as model
import models.controller.forecast.LstmAdapter as LstmAdapter


class LoadModelLstm():
    def __init__(self, clockfun, horizon_intsteps, data_name=None) -> None:
        self.clockfun = clockfun
        self.horizon_intsteps = horizon_intsteps

        # Load the public holiday calendar
        public_holidays_dict = holidays.CountryHoliday('DE', years=range(2013, 2014))
        self.public_holidays_timestamps = [pd.Timestamp(date, tzinfo=pytz.timezone('Europe/Vienna')) 
                                            for date in public_holidays_dict.keys()]

        self.lstmAdapter = LstmAdapter.LstmAdapter(
                    self.public_holidays_timestamps, 
                    add_tda_features=False, 
                    addLaggedPower=True, 
                    shuffle_data=False, 
                    seed=0,
                    prediction_rate = pd.Timedelta(days=0, hours=1),    # Choose a generating element of the group Z_24.​
                    prediction_horizon = pd.Timedelta(days=0, hours=23, minutes=0),
                    sampling_time = pd.Timedelta(hours=1, minutes=0),
                    prediction_history = pd.Timedelta(days=1, hours=0))

    def train(self, historic_load, weatherData):
        
        # Bring the aggregated characteristic power profiles to the format needed by the model
        #
        X, Y = self.lstmAdapter.transformData(historic_load, weatherData)

        # Train the model
        #
        optimizer = model.Optimizer(maxEpochs=10, set_learning_rates=[0.015, 0.005, 0.003, 0.002, 0.001, 0.001, 0.001])
        self.myModel = model.Model(optimizer, reg_strength = 0.00, shape=X['all'].shape[1:])
        self.myModel.compile(metrics=[])        
        self.path_to_pretrained_model = pathlib.Path('models/controller/model/', 'lstm_pretrain_weights.h5')
        self.myModel.load_weights(self.path_to_pretrained_model)
        self.myModel.model.fit( x=X['all'], y=Y['all'],
                                # batch_size=X['all'].shape[0],
                                epochs=optimizer.maxEpochs,
                                shuffle=True,
                                verbose=0, # 2 (changed by seva)
                                callbacks=[optimizer.get_lr_callback()])
        
        # self.historic_load = historic_load  # store the historic load, for later use during predict() (should net be needed, is passed directly to predict, changed by SEVA)

    def predict(self, inttime, weatherData, historic_load_data):
        
        # Test, if the weather is up-to-date
        #
        act_weather_timestamp = weatherData.index[-1]
        current_timestep = self.clockfun(inttime)
        assert act_weather_timestamp == current_timestep, \
            "Weather time must be up-to-date!" \
            f" Instead it is: {act_weather_timestamp} (!= {current_timestep})"

        # Predict
        #
        X = self.lstmAdapter.get_X_for_prediction(weatherData, historic_load_data)  # self.historic_load) changed by SEVA if we optimize only once a week, we dont have the 7 day lag power otherwise (power is stored at 00:00 and sent at 12:00)
        predicted_profile_raw = self.myModel.model.predict(X, verbose=0)
        predicted_profile = np.squeeze(self.lstmAdapter.deNormalizeY(predicted_profile_raw))

        assert predicted_profile.shape == (self.horizon_intsteps,), \
            "Prediction shape incorrect!" \
            f" Instead it is: {predicted_profile.shape} (!= ({self.horizon_intsteps},))"
        
        return predicted_profile
