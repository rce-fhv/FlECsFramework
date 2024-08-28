import pandas as pd
import numpy as np
import datetime


# Bring the data into the data format needed by the model
#
class LstmAdapter:

    def __init__(self,
                 public_holidays,
                 train_size = 263,
                 dev_size = 0,
                 add_tda_features = False,
                 addLaggedPower=False,
                 shuffle_data=True,
                 seed=None, 
                 use_persistent_entropy = True,
                 tda_forecast=None,
                 prediction_rate = pd.Timedelta(days=1),
                 prediction_horizon = pd.Timedelta(days=0, hours=23, minutes=0),
                 sampling_time = pd.Timedelta(hours=1, minutes=0),
                 prediction_history = pd.Timedelta(days=1, hours=0),
                 ):

        self.prediction_rate = prediction_rate
        self.prediction_horizon = prediction_horizon
        self.sampling_time = sampling_time
        self.prediction_history = prediction_history
        self.public_holidays = public_holidays
        self.addLaggedPower = addLaggedPower
        self.shuffle_data = shuffle_data
        self.train_size = train_size
        self.dev_size = dev_size       
        self.add_tda_features = add_tda_features
        self.use_persistent_entropy = use_persistent_entropy
        self.tda_forecast = tda_forecast 

        # Fix the random-seed for reproducibility
        if seed != None:
            np.random.seed(seed)

        # Optionally TDA Feature may be enabled
        if self.add_tda_features == True:                    
            from gtda.time_series import SingleTakensEmbedding
            from gtda.homology import VietorisRipsPersistence
            from gtda.diagrams import PersistenceEntropy

            self.embedder = SingleTakensEmbedding(
                parameters_type="fixed",
                time_delay=1,
                dimension=3,
                stride=1
            )
            self.VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
            self.PE = PersistenceEntropy()

        # Set the maximum power lag needed by the input features
        if self.add_tda_features == True:
            self.max_needed_power_lag = pd.Timedelta(days=22, hours=0)
        elif self.addLaggedPower == True:
            self.max_needed_power_lag = pd.Timedelta(days=22, hours=0)
        else:
            self.max_needed_power_lag = pd.Timedelta(days=0, hours=0)

    def transformData(self, 
                      powerProfiles, 
                      weatherData, 
                      training=True,
                      first_prediction_clocktime = datetime.time(0, 0),
                      ):

        # Downsample the profiles to 1h frequency
        powerProfiles = self.downsample(powerProfiles)

        # Get the first and last available timestamps
        self.first_prediction_date = self.getFirstPredictionTimestamp(powerProfiles, first_prediction_clocktime)
        self.last_available_datetime = powerProfiles.index[-1]

        # Convert the power timeseries to a nd-array with format (nr_of_batches, timesteps, outputs)
        Y_all = self.formattingY(powerProfiles)

        # Convert the input features to a nd-array with format (nr_of_batches, timesteps, features)
        X_all = self.formattingX(weatherData, powerProfiles, training=training)

        # Split up the data into train, dev, test and modeldata
        X, Y = self.splitUpData(X_all, Y_all)

        return X, Y

    def get_X_for_prediction(self, weatherData, powerProfiles):

        # No power profile is given, like e.g during inference without any lagged power feature.
        #            
        self.first_prediction_date = weatherData.index[-1]
        self.last_available_datetime = weatherData.index[-1] + self.prediction_horizon

        # Convert the input features to a nd-array with format (nr_of_batches, timesteps, features)
        X = self.formattingX(weatherData, powerProfiles, training=False)

        return X
        
    def getFirstPredictionTimestamp(self, powerProfiles, first_prediction_clocktime):

        # Calculate the first possible prediction timestamp
        first_timestamp = powerProfiles.index[0] + max(self.prediction_history, self.max_needed_power_lag)

        # Choose a prediction datetime, which is on the same day as the 'first_timestamp'.
        target_timestamp = pd.Timestamp.combine(first_timestamp.date(), first_prediction_clocktime) \
                    .tz_localize(first_timestamp.tzinfo)

        # Check if the calculated timestamp is before or after the target time
        if target_timestamp < first_timestamp:
            first_prediction_timestamp = target_timestamp + pd.Timedelta(days=1)
        else:
            first_prediction_timestamp = target_timestamp

        return first_prediction_timestamp


    # Downsample all signals, i.e. from 1min to 1h sample period,
    # by using the 'resample()' methode provided by pandas.
    #
    def downsample(self, data, downsamplingFactor='1'):
        
        data_PL_downsampled = data.resample(downsamplingFactor + 'h').mean()

        return data_PL_downsampled

    # Convert the input data to the LSTM format.
    # For more informations regarding the shape see LSTM design for this project.
    #
    def formattingX(self, weatherData, powerProfiles=None, training=True):

        batch_id = 0
        next_prediction_date = self.first_prediction_date

        # Calculate/define the shape of X
        nr_of_features = 11
        if self.add_tda_features == True:
            nr_of_features += 3
        if self.tda_forecast is not None:
            nr_of_features += 1
        if self.addLaggedPower == True:
            nr_of_features += 3

        # Define the number of weather features
        if weatherData is None:
            num_of_weather_features = 4 # Default weather features
        else:
            num_of_weather_features = weatherData.shape[1]
        nr_of_features += num_of_weather_features

        seq_start_time = self.first_prediction_date - self.prediction_history
        seq_end_time = self.first_prediction_date + self.prediction_horizon
        nr_of_timesteps = len(pd.date_range(start=seq_start_time, end=seq_end_time, freq=self.sampling_time))
        X_all_data = np.zeros(shape=(0, nr_of_timesteps, nr_of_features))

        while next_prediction_date + self.prediction_horizon <= self.last_available_datetime:

            # Add a new column to the X array
            new_values = np.zeros(shape=(1, nr_of_timesteps, nr_of_features))
            X_all_data = np.concatenate((X_all_data, new_values), axis=0)

            # Define the current time range
            start_datetime = next_prediction_date - self.prediction_history
            end_of_prediction = next_prediction_date + self.prediction_horizon
            total_input_range = pd.date_range(start=start_datetime, end=end_of_prediction, freq=self.sampling_time)

            # Get the current weekday indices [0 ... 6] of all nr_of_timesteps.
            # The shape of the following variable is (nr_of_timesteps, 1).
            weekday_numbers = total_input_range.weekday.values
            
            # Identify public holidays and replace that day with Sunday
            public_holiday_indices = total_input_range.floor("D").isin(self.public_holidays)
            weekday_numbers[public_holiday_indices] = 6

            # Create a one-hot encoding array with shape (nr_of_timesteps, 7).
            one_hot_encoding = np.eye(7)[weekday_numbers]
            index = 7
            X_all_data[batch_id, :, :index] = one_hot_encoding

            # Convert clock_time to cyclical features
            hour_sin = np.sin(2 * np.pi * total_input_range.hour / 23.0)
            hour_cos = np.cos(2 * np.pi * total_input_range.hour / 23.0)
            X_all_data[batch_id, :, index]  = hour_sin
            index += 1
            X_all_data[batch_id, :, index]  = hour_cos
            index += 1

            # Convert day-of-year to cyclical features
            day_of_year_sin = np.sin(2 * np.pi * total_input_range.day_of_year / 365)
            day_of_year_cos = np.cos(2 * np.pi * total_input_range.day_of_year / 365)
            X_all_data[batch_id, :, index]  = day_of_year_sin
            index += 1
            X_all_data[batch_id, :, index]  = day_of_year_cos
            index += 1

            # Optionally add Topological Feature
            if self.add_tda_features == True:

                # Get high dimensional data
                prev_days = []
                for day in range(1, 4):
                    start = next_prediction_date - pd.Timedelta(days=day*7, hours=0)
                    end = start + pd.Timedelta(days=0, hours=23)
                    prev_day = powerProfiles.loc[start:end]
                    prev_days.append(np.array(prev_day.values))

                use_takens_embeding = True
                if use_takens_embeding == True:
                    powerProfile_short_history = np.concatenate(prev_days, axis=0)
                    ph_input = self.embedder.fit_transform(powerProfile_short_history)
                else:
                    prev_days.append(np.array(range(24))) # add clock time
                    ph_input = np.stack(prev_days, axis=1)

                # Calculate persistence diagram
                act_ph_diagram = self.VR.fit_transform(ph_input[np.newaxis,:,:])

                if self.use_persistent_entropy == True:

                    # Get persistence entropy
                    features = self.PE.fit_transform(act_ph_diagram)

                else: 

                    # Calculate persistence lifetime
                    differences = act_ph_diagram[0,:, 1] - act_ph_diagram[0, :, 0]

                    # Group differences according to persistence dimension
                    groups = {}
                    for i, value in enumerate(act_ph_diagram[0, :, 2]):
                        key = int(np.round(value))
                        if key not in groups:
                            groups[key] = []
                        groups[key].append(differences[i])

                    # Get max persistence
                    features = np.zeros(shape=3)
                    for key, value in groups.items():
                        features[key] = np.max(value)

                X_all_data[batch_id, :, index:index+3]  = features
                index += 3

            if self.tda_forecast is not None:
                if batch_id > 0:
                    X_all_data[batch_id, :24, index]  = self.tda_forecast[batch_id-1, :]
                X_all_data[batch_id, 24:, index]  = self.tda_forecast[batch_id, :]
                index += 1

            # Optionally add lagged profiles
            if self.addLaggedPower == True:
                prev_days = []
                for day in range(1, 4):
                    start = next_prediction_date - pd.Timedelta(days=day*7, hours=0)
                    end = start + pd.Timedelta(days=0, hours=23)
                    prev_day = powerProfiles.loc[start:end]                    
                    X_all_data[batch_id, -len(prev_day):, index]  = np.array(prev_day.values)
                    index += 1

            # If available: Add past weather measurmenents to the LSTM input
            if weatherData is not None:
                weatherData_slice = weatherData.loc[start_datetime:next_prediction_date]
                weather_seq_len = weatherData_slice.shape[0]
                for feature in weatherData_slice.columns:
                    X_all_data[batch_id, :weather_seq_len, index]  = weatherData_slice[feature][:]
                    index += 1
            else:
                X_all_data[batch_id, :, index:num_of_weather_features]  = 0.0
                index += num_of_weather_features

            # Go to the next prediction (= batch)
            next_prediction_date += self.prediction_rate
            batch_id += 1

        # Normalize LSTM input data
        X_all_data = self.normalizeX(X_all_data, training=training)

        return X_all_data
    
    # Convert the given power profiles to the LSTM format.
    # For more informations regarding the shape see LSTM design for this project.
    #
    def formattingY(self, df):
        
        batch_id = 0
        next_prediction_date = self.first_prediction_date

        # Calculate/define the shape of Y
        seq_end_time = self.first_prediction_date + self.prediction_horizon
        nr_of_timesteps = len(pd.date_range(start=self.first_prediction_date, end=seq_end_time, freq=self.sampling_time))
        Y_all_data = np.zeros(shape=(0, nr_of_timesteps, 1))

        while next_prediction_date + self.prediction_horizon <= self.last_available_datetime:
            
            # Add a new column to the Y array
            new_values = np.zeros(shape=(1, nr_of_timesteps, 1))
            Y_all_data = np.concatenate((Y_all_data, new_values), axis=0)

            # Get values within the specified time range
            end_prediction_horizon = next_prediction_date + self.prediction_horizon
            demandprofile_slice = df.loc[next_prediction_date:end_prediction_horizon]
            
            # Set all target power values
            Y_all_data[batch_id, :, 0] = demandprofile_slice

            # Go to the next prediction (= batch)
            next_prediction_date += self.prediction_rate
            batch_id += 1

        # Do a min/max normalisation of the target values (to [0 ... 1])
        Y_all_data = self.normalizeY(Y_all_data, training=True)

        return Y_all_data

    # Z-Normalize the input data of the LSTM.
    #
    def normalizeX(self, X, training=False):

        if training:
            # Estimate the mean and standard deviation of the data during training
            self.meanX = np.mean(X, axis=(0, 1))
            self.stdX = np.std(X, axis=(0, 1))
        
        if np.isclose(self.stdX, 0).any():
            # Avoid a division by zero (which can occur for constant features)
            self.stdX = np.where(np.isclose(self.stdX, 0), 1e-8, self.stdX)

        X_normalized = (X - self.meanX) / self.stdX

        return X_normalized

    # Undo z-normalization
    #
    def deNormalizeX(self, X):

        X_denormalized = (X * self.stdX) + self.meanX

        return X_denormalized

    # Normalize the output data of the LSTM.    
    #
    def normalizeY(self, Y, training=False):

        if training:
            # Estimate the standard deviation of the data during training
            self.stdY = np.std(Y)
        
        if np.isclose(self.stdY, 0):
            assert False, "Normalization leads to division by zero."

        Y_normalized = Y / self.stdY

        return Y_normalized

    # Undo normalization
    #
    def deNormalizeY(self, Y):

        Y_denormalized = Y * self.stdY

        return Y_denormalized
    
    # Split up the data into train, dev, test and modeldata
    #
    def splitUpData(self, X_all, Y_all):

        # Optionally shuffle all indices
        total_samples = X_all.shape[0]
        self.shuffeled_indices = np.arange(total_samples)
        if self.shuffle_data == True:
            np.random.shuffle(self.shuffeled_indices)

        # Split up data
        X, Y = {}, {}
        X['train'] = X_all[self.shuffeled_indices[:self.train_size]]
        X['dev'] = X_all[self.shuffeled_indices[self.train_size:self.train_size+self.dev_size]]
        X['test'] = X_all[self.shuffeled_indices[self.train_size+self.dev_size:]]
        X['all'] = X_all[:]

        Y['train'] = Y_all[self.shuffeled_indices[:self.train_size]]
        Y['dev'] = Y_all[self.shuffeled_indices[self.train_size:self.train_size+self.dev_size]]
        Y['test'] = Y_all[self.shuffeled_indices[self.train_size+self.dev_size:]]
        Y['all'] = Y_all[:]

        return X, Y
    
    # Return the unshuffled index in all data that corresponds to the given
    # dataset_tye and index.
    #
    def getUnshuffeledIndex(self, dataset_type, index):

        # Shuffled data
        if dataset_type == 'train':
            unshuffled_index = self.shuffeled_indices[index]
        elif dataset_type == 'dev':
            unshuffled_index = self.shuffeled_indices[index + self.train_size]
        elif dataset_type == 'test':
            unshuffled_index = self.shuffeled_indices[index+self.train_size+self.dev_size]
        else:
            assert False, "Unexpected 'dataset_type' parameter received."

        return unshuffled_index 
    
    # Return the prediction date that corresponds to the given
    # dataset_tye and index.
    #
    def getStartDateFromIndex(self, dataset_type, index):        

        if dataset_type != 'all':
            index = self.getUnshuffeledIndex(dataset_type, index)

        return self.first_prediction_date + index * self.prediction_rate
    
    # Return the dataset-type (train, test, ...) from the given unshuffeled index
    #
    def getDatasetTypeFromIndex(self, unshuffeled_index):

        shuffled_index = np.where(self.shuffeled_indices == unshuffeled_index)[0]
        dataset_type = ''

        if shuffled_index < self.train_size:
            dataset_type = 'train'
        elif shuffled_index < self.train_size + self.dev_size:
            dataset_type = 'dev'
        else:
            dataset_type = 'test'

        return dataset_type

if __name__ == '__main__':
    pass

