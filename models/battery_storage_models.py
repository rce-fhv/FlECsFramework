"""
This module contains the Lithium Ion battery storage model for the Energy Community scenario.
"""

class StorageModelWithEfficiency:
    def __init__(self, capacity_0=2500 * 3600, **params) -> None:
        '''
        Storage Model with simple charge and discharge efficiencys
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

        # outputs / states
        self.capacity         = capacity_0  # J
        self.actual_power     = 0 # None  # W
        self.power            = 0 # None  # W

    def step(self, power, time):
        '''
        Power: power>0... discharging, power<0... charging
        time: inttime for debugging
        '''
        self.power = power # save for mosaik
        
        # performe sanity check...
        assert self.min_capacity <= self.capacity <= self.max_capacity, 'something is really off here!'

        power = min(self.max_power_discharge, max(power, -self.max_power_charge))  # maximum / minimum power constraint

        # calculate potential new capacity
        if power < 0: ############################################################################################################
            new_capacity = self.capacity - power * self.delta_t * self.eta_i
        else:
            new_capacity = self.capacity - power * self.delta_t / self.eta_o

        # check limits and adjust
        if new_capacity < self.min_capacity: # storage empty
            self.actual_power = (self.capacity - self.min_capacity) / self.delta_t * self.eta_o
            self.capacity = self.min_capacity
        elif new_capacity > self.max_capacity: # storage full
            self.actual_power = (self.capacity - self.max_capacity) / self.delta_t / self.eta_i
            self.capacity = self.max_capacity
        else:
            self.actual_power = power
            self.capacity = new_capacity

        # print('control power: ', power, 'actual power: ', self.actual_power, 'difference: ', self.actual_power-power)
        return self.actual_power, self.capacity, # self.power
    

class StorageModelWithSelfdischarge:
    def __init__(self, capacity_0=2500 * 3600, **params) -> None:
        '''
        Storage Model with simple charge and discharge efficiencys
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
        self.self_discharge_rate = params.pop('self_discharge_rate',2.e-8) # 1/s

        # outputs / states
        self.capacity         = capacity_0  # J
        self.actual_power     = 0 # None  # W
        self.power            = 0 # None  # W

    def step(self, power, time):
        '''
        Power: power>0... discharging, power<0... charging
        time: inttime for debugging
        '''
        self.power = power # save for mosaik

        # performe sanity check...
        assert self.min_capacity <= self.capacity <= self.max_capacity, 'something is really off here!'

        power = min(self.max_power_discharge, max(power, -self.max_power_charge))  # maximum / minimum power constraint

        # calculate potential new capacity
        if power < 0: ################################################################################################
            new_capacity = self.capacity - power * self.delta_t * self.eta_i
        else:
            new_capacity = self.capacity - power * self.delta_t / self.eta_o

        # check limits and adjust
        if new_capacity < self.min_capacity: # storage empty
            self.actual_power = (self.capacity - self.min_capacity) / self.delta_t * self.eta_o
            self.capacity = self.min_capacity
        elif new_capacity > self.max_capacity: # storage full
            self.actual_power = (self.capacity - self.max_capacity) / self.delta_t / self.eta_i
            self.capacity = self.max_capacity
        else:
            self.actual_power = power
            self.capacity = new_capacity

        # apply self discharge
        self.capacity = self.capacity*(1-self.self_discharge_rate*self.delta_t)

        # print('control power: ', power, 'actual power: ', self.actual_power, 'difference: ', self.actual_power-power)
        return self.actual_power, self.capacity
    