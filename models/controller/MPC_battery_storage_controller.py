import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from copy import copy

# move this thing to utility and get the damn relative imports to work 
def plot_time_discrete(self, x, y, *args, **kwargs):
    '''
    Plots either a step function in between a time index if len(x) == len(y)+1 or a continuous plot when len(x) == len(y)
    x: time values of the plot
    y: corersponding y values in between two x values
    
    visualization...
    either plot step plot
         y1        y2
    x1--------x2--------x3

    or continuous plot
    y1        y2        y3
    x1--------x2--------x3
    '''
    if len(x) == len(y) +1:
        # where is specified here and can lead to wrong behaviour
        kwargs.pop('where', None)

        # extend the y values
        y = list(y)
        y = y + [y[-1]]
        self.step(x, y, where='post', *args, **kwargs)
    elif len(x) == len(y):
        self.plot(x, y, *args, **kwargs)
    else:
        raise ValueError('Length of x any y does not match, it can either be len(x) == len(y) or len(x) == len(y)+1')
    
mpl.axes.Axes.plot_time_discrete = plot_time_discrete


class MPCBatteryStorageController:
    def __init__(self, clockfun, delta_t, initial_controll_power, milp_battery_model, predictor, warmup_time:pd.Timedelta) -> None:
        self.delta_t = delta_t # s
        self.controll_power = initial_controll_power
        self.clockfun = clockfun
        self.actual_storage_power_history = pd.Series([])
        self.warmup_time = warmup_time 

        self.residual_power_predictor = predictor
        self.milp_battery_model       = milp_battery_model

        self.prediction_available = False # run the controler only if there is a prediction available

        # stored historic data for prediction
        self.pv_power = pd.Series([])
        self.historic_weather_data = pd.DataFrame([], dtype=float, columns=['temp_air', '0', '1', '2']) # TODO:This could be a series, if the lstm accepts it but has to contain 4 columns for now
        self.historic_residual_load = pd.Series([])
        self.historic_residual_load.index.freq = f'{self.delta_t}s' # set frequency for residual load prediction (needed for shifting the index of the persistence forecast)

        # prediction data (for storing and visalizing)
        self.current_prediction       = pd.Series([])

    def step(self, time, capacity_sorage_tminus1, power_storage_tminus1, historic_load_data, pv_power, temp_air):
        # send only 2 months of data to the predictor for comparability
        # start control logic only after 2 months of data availability, otherwise return zero
        datetime = self.clockfun(time)
        
        ### DATA HANDLING###
        # store pv power and temp air for later use with the predictors
        self.pv_power.at[datetime] = pv_power
        self.historic_weather_data.loc[datetime, ['temp_air', '0', '1', '2']] = [temp_air, 0, 0, 0] # make up fake weather data for the lstm, it expects four features but currently form the pv we only have 1
        # store actual storage power for later use with the predictors
        self.actual_storage_power_history[self.clockfun(time-1)] = power_storage_tminus1

        # store residual load if it is available for later use with the predictors
        if historic_load_data is not None:          
            historic_residual_load = (historic_load_data.sum(axis=1) - self.actual_storage_power_history).dropna() # sum up the loads, historic residual load contains the battery power, so it needs to be subtracted
            self.historic_residual_load = pd.concat([self.historic_residual_load, historic_residual_load[~historic_residual_load.index.isin(self.historic_residual_load.index)]], axis='index', verify_integrity=True) # TODO: remove verify integrity for speed!  TODO: Consider if copy=False in concat works (faster)

        ### Update prediction model ###
        # make the predictio only if there is enough data available for training of the prediction model (warmup time) and only if new data is available 
        if (not self.historic_residual_load.empty 
            and datetime - self.historic_residual_load.index[0] >=self.warmup_time 
            and historic_load_data is not None
            and (not getattr(self.residual_power_predictor, 'train_weekly', False) or datetime.weekday()==1)): # run lstm only once a week, when new data is available
            # print(f'updating prediction model now {datetime}')
            self.residual_power_predictor.update_prediction_model(historic_residual_load=self.historic_residual_load, historic_power_generation=self.pv_power, historic_weather_data=self.historic_weather_data) # TODO maybe cut pv power and historic_weather_data, to not sent the whole data ech cycle 
            self.prediction_available = True # this will be set true on the first time the prediction is called

        ### Controller ###
        # run the controller only if a valid prediction is available
        if self.prediction_available: # this ise set true on the first time the prediction is called
            # make and get prediction
            # print(f'updating prediction now {datetime}')
            self.residual_power_predictor.update_prediction(time, historic_weather_data=self.historic_weather_data, historic_power_generation=self.pv_power, historic_residual_load=self.historic_residual_load)
            self.current_prediction = self.residual_power_predictor.get_current_prediction(time) # currently this needs to be called right after update_prediction for most predictors)
            # optimize model
            self.storage_controll_power_horizon, self.storage_controll_capacity_horizon, self.storage_controll_capacity1_horizon = self.milp_battery_model.solve(  # "-"" !!!
                                            capacity_tminus1=capacity_sorage_tminus1,
                                            power_storage_0=self.controll_power, # "-" !!!
                                            residual_power_prediction=self.current_prediction, make_full_output=True) # disable this for performance later!
            # self.milp_battery_model.test_plot_solve(time=time)
            # plt.show(block=False)
            # if self.controll_power != self.storage_controll_power_horizon.iloc[0]: # test of the power is correct, this is only applicable for perfect prediction
            #     print(f'here it is!!!, {datetime} {self.controll_power} != {self.storage_controll_power_horizon.iloc[0]}')
            next_controll_power = self.storage_controll_power_horizon.iloc[1] # take second value (first value is from previous step)    !!!!!!!!!!!!!!!!!!!!!!!!
        else:
            self.storage_controll_power_horizon      = pd.Series([])
            self.storage_controll_capacity_horizon   = pd.Series([])
            self.storage_controll_capacity1_horizon  = pd.Series([])
            next_controll_power = 0
        
        self.controll_power = copy(next_controll_power)
        return datetime, next_controll_power


class MILPBatteryModel:
    def __init__(self, **params) -> None:
        '''
        params: delta_t, max_capacity, min_capacity, eta_i, eta_o, max_power_charge, max_power_discharge
        '''
        # model parameters
        self.delta_t             = params.pop('delta_t',             3600) # s
        self.max_capacity        = params.pop('max_capacity',50000 * 3600) # J
        self.min_capacity        = params.pop('min_capacity',           0) # J
        self.eta_i               = params.pop('eta_i',               0.95) #
        self.eta_o               = params.pop('eta_o',               0.95) #
        self.max_power_charge    = params.pop('max_power_charge',   10000) # W
        self.max_power_discharge = params.pop('max_power_discharge',10000) # W

        self.objective_key       = params.pop('objective_key', 'selfconsumption')
        
        # Data for plotting (only updated if make_full_output=True)
        self.output_df_steps           = None
        self.output_df_times           = None
        self.capacity_tminus1          = 0 # None
        # self.capacity_1                = 0 # None
        self.power_storage_0           = None
        self.residual_power_prediction = None

        # Plot setup
        self.fig, self.ax = plt.subplots(4, 1, sharex=True)
        # plt.close(self.fig) #  do not show figure instantly

    def solve(self, capacity_tminus1, power_storage_0:float, residual_power_prediction:pd.Series, make_full_output=False):
        '''
        Solve the optimization problem for one time horizon\n
        step      0        1        2     
        time  |--------|--------|--------|
              0        1        2        3
              |< now  >|<     future    >|
              |< opt  >|<  set >| 

        capacity_tminus1: float; current state of charge
        power_storage_0: float; current power of the storage
        residual_power_prediction: pd.Series; production - consumption for all steps (starting at index 0!)
        make_full_outputf: if True, the dataframe 'output_df_times' and 'output_df_steps' containing the complete output and the input parameters are created (for testing and visualizing)
        '''
        # for esthetic reasons and potentially performance, 
        # this model could be implemented as an absract model, 
        # so the model declaration would not have to be done repetetively
        
        # print(self.capacity_1, capacity_tminus1, power_storage_0, (capacity_tminus1-self.capacity_tminus1)/self.delta_t)

        # perform data checks
        assert self.min_capacity <= capacity_tminus1 <= self.max_capacity, 'capacity_tminus1 out of bounds'
        assert -self.max_power_charge <= power_storage_0 <= self.max_power_discharge, f'power_storage_0 = {power_storage_0} out of bounds [{-self.max_power_charge}, {self.max_power_discharge}]'
        assert len(residual_power_prediction) >= 2, 'residual_power_prediction time horizon not long enough for optimization'
        assert not residual_power_prediction.hasnans, 'Residual Power Prediction contains Nan values, can not solve!'

        # if storage would be too empty or to full to discharge / charge power_storage_0 (due to e.g. self discharge) adjust limits
        if power_storage_0 > 0:
            if (capacity_tminus1 - power_storage_0*self.delta_t/self.eta_o) < self.min_capacity:
                # print(f'adjusting power from {power_storage_0} to {(capacity_tminus1-self.min_capacity)*self.eta_o/self.delta_t}')
                power_storage_0 = (capacity_tminus1-self.min_capacity)*self.eta_o/self.delta_t
        else:
            if (capacity_tminus1 - power_storage_0*self.delta_t*self.eta_i) > self.max_capacity:
                # print(f'adjusting power from {power_storage_0} to {(capacity_tminus1-self.max_capacity)/self.eta_i/self.delta_t}')
                power_storage_0 = (capacity_tminus1-self.max_capacity)/self.eta_i/self.delta_t
                
        #<= self.max_capacity+0.001, f'power and capacity of first time step lead to infeasibility (capacity_1 out of bounds: {self.min_capacity/3600} <= {(capacity_tminus1 - power_storage_0*self.delta_t)/3600} <= {self.max_capacity/3600})'
        
        # Convert residual Power Prediction to numpy array and save index for safety
        self.datetime_index = residual_power_prediction.index
        residual_power_prediction = residual_power_prediction.values.ravel()

        # pyomo setup
        m = pyo.ConcreteModel()
        opt = pyo.SolverFactory('glpk')
        
        # Indices
        # m.steps = pyo.RangeSet(0, len(residual_power_prediction)-1) # discrete time steps for the time horizon
        m.steps = pyo.RangeSet(0, len(residual_power_prediction)-1) # discrete time steps for the time horizon    ################## ACHTUNG
        m.times = pyo.RangeSet(-1, len(residual_power_prediction)-1)  # discrete time points for the time horizon   ################## ACHTUNG
        # m.steps_from_one = pyo.RangeSet(1, len(residual_power_prediction))  # discrete time steps for the time horizon, excluding step one (for power constraints)
        
        # Variables
        m.power_storage_charge    = pyo.Var(m.steps, domain=pyo.NonNegativeReals) # W
        m.power_storage_discharge = pyo.Var(m.steps, domain=pyo.NonNegativeReals) # W
        m.power_grid_i            = pyo.Var(m.steps, domain=pyo.NonNegativeReals) # W
        m.power_grid_o            = pyo.Var(m.steps, domain=pyo.NonNegativeReals) # W
        m.b_charge                = pyo.Var(m.steps, domain=pyo.Binary) # 0/1
        m.b_discharge             = pyo.Var(m.steps, domain=pyo.Binary) # 0/1
        m.capacity                = pyo.Var(m.times, bounds=(self.min_capacity, self.max_capacity)) # J

        # Constraints
        ## Energy Balance for grid connection
        # m.grid_energy_balance = pyo.Constraint(m.steps, 
        #                 rule = lambda m, i: 
        #                 m.power_storage_discharge[i]- m.power_storage_charge[i] 
        #                 + m.power_grid_i[i] - m.power_grid_o[i] 
        #                 + residual_power_prediction[i] == 0 )
        m.grid_energy_balance = pyo.Constraint(m.steps, 
                        rule = lambda m, i: 
                        m.power_storage_discharge[i]- m.power_storage_charge[i] 
                        + m.power_grid_i[i] - m.power_grid_o[i] 
                        + residual_power_prediction[i] == 0 ) ########################## ACHTUNG

        # ## Storage Model
        # m.storage_energy_balance   = pyo.Constraint(m.steps, 
        #                 rule = lambda m, i: m.capacity[i+1] == m.capacity[i] 
        #                 + m.power_storage_charge[i]*self.eta_i*self.delta_t 
        #                 - m.power_storage_discharge[i]*self.delta_t/self.eta_o)
        
        m.storage_energy_balance   = pyo.Constraint(m.steps,  ############### AAAAAAAAAACCCCCCCHHHHHHHHHHHHHHTTTTTTTTTUUUUUUUUUUUUUUUNNNNNNNGGGGGGGGGGGG
                        rule = lambda m, i: m.capacity[i] == m.capacity[i-1] 
                        + m.power_storage_charge[i]*self.eta_i*self.delta_t 
                        - m.power_storage_discharge[i]*self.delta_t/self.eta_o)

        ## do not charge and discharge at the same time
        m.charge_discharge_limit_1   = pyo.Constraint(m.steps, 
                        rule = lambda m, i: m.power_storage_charge[i] <= self.max_power_charge * m.b_charge[i])
        m.charge_discharge_limit_2   = pyo.Constraint(m.steps, 
                        rule = lambda m, i: m.power_storage_discharge[i] <= self.max_power_discharge * m.b_discharge[i])
        m.charge_discharge_limit_3   = pyo.Constraint(m.steps, 
                        rule = lambda m, i: m.b_discharge[i] + m.b_charge[i] <= 1)
        
        ## start and end conditions
        m.storage_initail_capacity = pyo.Constraint(expr = m.capacity[-1] == capacity_tminus1)
        # m.storage_final_capacity   = pyo.Constraint(expr = m.capacity[m.times.at(-1)] == capacity_tminus1) ##################

        if power_storage_0 < 0:
            m.storage_initial_power_charge    = pyo.Constraint(expr = m.power_storage_charge[0] == -power_storage_0)  ########################## ACHTUNG
            m.storage_initial_power_discharge = pyo.Constraint(expr = m.power_storage_discharge[0] == 0) ########################## ACHTUNG
        else:
            m.storage_initial_power_discharge = pyo.Constraint(expr = m.power_storage_discharge[0] == power_storage_0) ########################## ACHTUNG
            m.storage_initial_power_charge    = pyo.Constraint(expr = m.power_storage_charge[0] == 0) ########################## ACHTUNG

        # Objective 
        if self.objective_key == 'selfconsumption':
            m.OBJ = pyo.Objective(rule = lambda m: sum(m.power_grid_o[i]  for i in m.steps))
        elif self.objective_key == 'selfsufficiency':
            m.OBJ = pyo.Objective(rule = lambda m: sum(m.power_grid_i[i]  for i in m.steps)) 
        elif self.objective_key == 'price_driven':
            m.OBJ = pyo.Objective(rule = lambda m: sum(m.power_grid_o[i] * 0.125 - m.power_grid_i[i] * 0.19 for i in m.steps), sense=-1)
        elif self.objective_key == 'min_max_power':
             m.max_power_i              = pyo.Var(domain=pyo.NonNegativeReals) # W
             m.max_power_o              = pyo.Var(domain=pyo.NonNegativeReals) # W
             m.steps_future             = pyo.RangeSet(1, len(residual_power_prediction)-1) # discrete time steps for the time horizon# for peak load reduction do not use the first step in the objective, as this might lead to a drift (ask WOPH!)
             m.limit_max_power_i        = pyo.Constraint(m.steps_future,
                                           rule = lambda m, i: m.power_grid_i[i] <=  m.max_power_i)
             m.limit_max_power_o        = pyo.Constraint(m.steps_future, 
                                           rule = lambda m, i: m.power_grid_o[i] <=  m.max_power_o)
             m.OBJ = pyo.Objective(rule = lambda m: m.max_power_i + m.max_power_o) 
             raise ValueError('This objective is not tested with new time indices!!!')
        else:  # * self.delta_t / 3600
            raise ValueError('No valid objective selected')
        # m.pprint()
        
        # solve the model 
        results = opt.solve(m)

        # check result for optimality
        
        assert results.solver.termination_condition == pyo.TerminationCondition.optimal, f'Optimization {results.solver.termination_condition}'

        if make_full_output:
            self.output_df_steps = pd.DataFrame.from_dict({
            'steps'                   : m.steps,
            'power_storage_charge'    : [m.power_storage_charge[i].value    for i in m.steps],
            'power_storage_discharge' : [m.power_storage_discharge[i].value for i in m.steps],
            'power_grid_i'            : [m.power_grid_i[i].value            for i in m.steps],           
            'power_grid_o'            : [m.power_grid_o[i].value            for i in m.steps],           
            'b_charge'                : [m.b_charge[i].value                for i in m.steps],               
            'b_discharge'             : [m.b_discharge[i].value             for i in m.steps]
            }).set_index('steps')

            self.output_df_times = pd.DataFrame({
            'times'                   : m.times,          
            'capacity'                : [m.capacity[t].value                for t in m.times]
            }).set_index('times')

        # self.capacity_tminus1          = capacity_tminus1
        # self.capacity_1                = m.capacity[0].value                      #############################################      check indices        
        # self.power_storage_0           = m.power_storage_discharge[1].value - m.power_storage_charge[1].value
        self.residual_power_prediction = residual_power_prediction
        self.objective_value = m.OBJ
        
        power_output = pd.Series(np.clip(np.array([m.power_storage_discharge[i].value - m.power_storage_charge[i].value for i in m.steps]), -self.max_power_charge, self.max_power_discharge), index=self.datetime_index) # clip for numerical inaccuracies from MILP
        capacity  = pd.Series([m.capacity[t].value for t in m.times][1:], index=self.datetime_index)
        capacity1 = pd.Series([m.capacity[t].value for t in m.times][:-1], index=self.datetime_index)
        return  power_output, capacity, capacity1
    

    def test_plot_solve(self, time=0):
        power_grid    = (self.output_df_steps['power_grid_i']-self.output_df_steps['power_grid_o']) # positiv from grid / to connection point
        power_storage = (self.output_df_steps['power_storage_discharge'] - self.output_df_steps['power_storage_charge']) # positive from storage / to connection point
        
        times         = self.output_df_times.index + time
        capacity      = self.output_df_times['capacity'] / 3600e3

        # self.fig, self.ax = plt.subplots(4, 1, sharex=True)

        self.ax[0].plot_time_discrete(times, self.residual_power_prediction/1e3, label=f'{time}', where='post')
        self.ax[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        self.ax[0].set_ylabel('$P_{EC, Pred} / kW$')
        self.ax[0].grid(visible=True)
        # self.ax[0].set_title('$P_{EC}$')
        # self.ax[0].legend()
    
        self.ax[1].plot_time_discrete(times, -power_grid/1e3, label='$P_{grid}$')  # , where='post')
        self.ax[1].plot_time_discrete(times, np.full(len(times), -power_grid.max()/1e3), label='$P_{grid}$')
        self.ax[1].plot_time_discrete(times, np.full(len(times), -power_grid.min()/1e3), label='$P_{grid}$')
        self.ax[1].set_ylabel('$P_{grid} / kW$')
        self.ax[1].grid(visible=True)
        # self.ax[1].set_title('$P_{grid}$')
        # self.ax[1].legend()

        self.ax[2].plot_time_discrete(times, power_storage/1e3, label='$P_{storage}$', where='post')
        self.ax[2].set_ylabel('$P_{storage} / kW$')
        self.ax[2].grid(visible=True)
        # self.ax[2].set_title('$P_{storage}$')
        # self.ax[2].legend()

        self.ax[3].plot_time_discrete(times, capacity, label='$E_{storage}$')
        self.ax[3].set_ylabel('$E_{storage} / kWh$')
        self.ax[3].grid(visible=True)
        # self.ax[3].set_title('$E_{storage}$')
        self.ax[3].axhline(self.max_capacity/ 3600e3)
        self.ax[3].axhline(0)
        # self.ax[3].legend()

        if time == 47:
            self.fig.set_size_inches(12, 12)
            self.fig.savefig(f'figures/test_plot_solve/plot_at_time_{time}.png') # , dpi=500)

# if __name__ == '__main__':
#     milp_model = MILPBatteryModel()

#     milp_model.solve(capacity_tminus1=500*3600, power_storage_0=-1000, residual_power_prediction=np.array([1000, 1000, -2000, 3000, -2000]), make_full_output=True)
#     milp_model.test_plot_solve()

    