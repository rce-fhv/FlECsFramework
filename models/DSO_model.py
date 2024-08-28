import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from utility.utility import TimeMatcher


class DSOModel:
    def __init__(self, clockfun, time_resolution) -> None:
        self.clock = clockfun

        # up to date values for storing internally
        self._node_data = pd.DataFrame([])
        self._electricity_ec = pd.DataFrame([])
        self._electricity_supplier = pd.DataFrame([])

        self._node_data_to_send = pd.DataFrame([])
        self._electricity_ec_to_send = pd.DataFrame([])
        self._electricity_supplier_to_send = pd.DataFrame([])

        # time delayed values for returning the date
        self.node_data            = None
        self.electricity_ec       = None
        self.electricity_supplier = None

        self.calc_share_timematcher                = TimeMatcher('****-**-** 00:00') # daily at 12:00
        self.send_data_to_ec_operator_timematcher  = TimeMatcher('****-**-** 12:00') # daily at 12:00
        # self.send_data_to_el_provider_timematcher  = TimeMatcher('****-**-01 00:00') # monthly at the first of the month

    def step(self, int_time, residential_loads:dict):
        # save data from nodes
        datetime = self.clock(int_time)
        self._node_data.loc[datetime, list(residential_loads.keys())] = list(residential_loads.values())
        # return_values = {}

        if datetime == self.calc_share_timematcher:
            # self.calc_share_dynamically()  # currently not needed (only later for analysis maybe)
            self._node_data_to_send            = self._node_data.loc[datetime-pd.Timedelta(hours=36):datetime].copy(deep=True)   #        
            # self._electricity_ec_to_send       = self._electricity_ec.copy(deep=True)       
            # self._electricity_supplier_to_send = self._electricity_supplier.copy(deep=True) 

        if datetime == self.send_data_to_ec_operator_timematcher:
            # cpopy the saved values to the output variables of the model when they should be made available
            self.node_data            = self._node_data_to_send.copy(deep=True)            
            # self.electricity_ec       = self._electricity_ec_to_send.copy(deep=True)       
            # self.electricity_supplier = self._electricity_supplier_to_send.copy(deep=True) 
        else:
            # when not sending time, set the output variables None
            self.node_data            = None
            self.electricity_ec       = None
            self.electricity_supplier = None

        # if datetime == self.send_data_to_el_provider_timematcher:
        #     self.calc_share_dynamically()
        #     return_values['e-sup_data'] = self.electricity_supplier
        
        # self.return_values = return_values

        # return return_values
    
    def calc_next_exec_time(self, current_int_time, timematcher):
        for i in range(current_int_time, current_int_time + 100000):
            next_int_time = -1
            if timematcher == self.clock(i):
                next_int_time = i
                return next_int_time
        raise ValueError('Exceeded next_exec_time check range without valid time match')

    def calc_share_dynamically(self):
        index                = self._node_data.index.difference(self._electricity_ec.index)
        node_data            = self._node_data.loc[index, :]


        consumption = node_data[node_data>0].sum(axis=1, skipna=True) # total consumption at any time step
        production = -node_data[node_data<0].sum(axis=1, skipna=True) # total production at any time step
        f = (production/consumption)  # factor
        electricity_ec = node_data.copy() # electricity traded in the Energy Community
        electricity_ec[node_data>0] = node_data[node_data>0].mul(f.apply(min, args=(1,)), axis=0)  # Calculate electricity traded in energy community when production < consumption
        electricity_ec[node_data<0] = node_data[node_data<0].mul(1/f.apply(max, args=(1,)), axis=0)  # Calculate electricity traded in energy community when production > consumption
        electricity_supplier = node_data - electricity_ec

        self._electricity_ec = pd.concat([self._electricity_ec, electricity_ec], axis='index')
        self._electricity_supplier = pd.concat([self._electricity_supplier, electricity_supplier], axis='index')


if __name__ == "__main__":
    dso_model = DSOModel()
    dso_model.step(1, {'h0':-2, 'h1':-2, 'h3':4, 'h4':6})
    dso_model.step(2, {'h0':-2, 'h1':-10, 'h3':4, 'h4':5})
    dso_model.step(3, {'h0':-2, 'h1':5, 'h3':4, 'h4':6})
    dso_model.calc_share_dynamically()
    print(dso_model.electricity_ec)
    print(dso_model.electricity_supplier)
