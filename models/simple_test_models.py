"""
This module contains models for testing the various frameworks.

"""
import numpy as np


class SimpleTestPVModel:
    """This model emulates a PV model.
    The model simply reads values from a static numpy array any time it is called
    """
    def __init__(self, size=2):
        """ size: size of PV system, scales the output"""
        self.delta_t = 1
        # generate fake PV output data 
        self._data = size*np.hstack([np.full(shape=3, fill_value=1), np.full(shape=7, fill_value=2)]) # 10x1
        
    def step(self, i):
        """Perform a simulation step by reading the data from the array
        negative power due to production"""
        return -self._data[i]
    

class SimpleTestHouseholdModel:
    """This Model implements a simple consumer model.
    The model simply reads values from a static numpy array any time it is called
    """
    def __init__(self, n_occupants=2):
        self.delta_t = 1
        # generate fake consumption data 
        # self._data = n_occupants*np.hstack([np.full(shape=7, fill_value=1), np.full(shape=3, fill_value=2)])
        self._data = n_occupants*np.array([1, 1, 1, 1, 1, 1, 1, 2, 0, 2])

    def step(self, i):
        """Perform a simulation step by reading the data from the array"""
        return self._data[i]
    

class SimpleTestGridModel:
    """This Model implements a simple consumer model.
    The model simply reads values from a static numpy array any time it is called
    """
    def __init__(self):
        self.delta_t = 1

    def step(self, i, powers=[]):
        """Perform a simulation step by summing up all inputs (e.g. calculating the remaining energy from or to the medium voltage grid)"""
        return sum(powers)
    

class SimpleTestFlexibilityModel:
    def __init__(self, max_capacity=5):
        """This model implements a simple storage model with a maximum capacity"""
        self.delta_t = 1
        self._capacity = 0 # actual capacity (SOC)
        self.max_capacity = max_capacity

    def step(self, i, control_input=0):
        """Perform a simulation step by adding the control input to the capacity if the storage is not empty or full"""
        new_capacity = self._capacity + control_input
        if new_capacity < 0:
            consumed_power = 0 - self._capacity
            self._capacity = 0
        elif new_capacity > self.max_capacity:
            consumed_power = self.max_capacity - self._capacity
            self._capacity = self.max_capacity
        else:
            consumed_power = control_input
            self._capacity = new_capacity
        return consumed_power, self._capacity
    

class SimpleTestFlexibilityWithAvailabilityModel(SimpleTestFlexibilityModel):
    def __init__(self, max_capacity=5, *args, **kwargs):
        """This model implements a simple storage model with a maximum capacity and an availability based on simple data"""
        super().__init__(max_capacity=max_capacity, *args, **kwargs)  # @ MOLU: hab schon langenichts mehr vererbt, ich hoffe das klappt so...
        self._availability = np.hstack([np.full(shape=7, fill_value=1), np.full(shape=3, fill_value=0)])

    def step(self, i, control_input=0):
        """Perform a simulation step by adding the control input to the capacity if the storage is not empty or full"""
        if self._availability[i]:
            consumed_power, capacity = super().step(i, control_input=control_input)
        else:
            consumed_power, capacity = 0, self._capacity
        
        return consumed_power, capacity
    

class SimpleTestFlexibilityControllerModel:
    def __init__(self):
        """This model implements a simple controller that controls a single flexibility based on the power difference in the power grid
        """
        pass

    def step(self, i, power_difference=0):
        """Perform a simulation step for the power controller based on the 
        if connected to the flexibility, it controlles for autarky"""
        return -power_difference
    

class SimpleTestCentralFlexibilityControllerModel:
    def __init__(self):
        """This model implements a simple controller that centrally manages decenral flexibility controllers based on the power difference in the power grid
        """
        self.communication_step = 0

    def step(self, i, power_difference=0, messages_from_controllers=''):
        """Perform a simulation step for the power controller (repeat all steps in one step) """
        if self.communication_step == 0:
            # send availability request to decentral controllers
            self.communication_step = 1
            message =  'request_availability'
            proceede = False
        elif self.communication_step == 1:
            # recieve availability  from controllers
            self.communication_step = 2
            n_available_controllers = sum(messages_from_controllers)
            if n_available_controllers > 0:
                message = power_difference / n_available_controllers
            else:
                message = 0
            proceede = False
        elif self.communication_step == 2:
            # finish step
            self.communication_step = 0
            proceede = True # tell the simulator to continue with the next time step
            message = 'done'
        return proceede, message
    
    
class SimpleTestDecentralFlexibilityControllerModel:
    def __init__(self):
        """This model implements a simple decentral controller that centrally manages flexibility controllers based on the interaction with the central controller
        """
        self._availability_data = np.hstack([np.full(shape=7, fill_value=1), np.full(shape=3, fill_value=0)])
        self._control_power = 0

    def step(self, i, message_from_central_controller=''):
        """Perform a simulation step for the power controller (repeat all steps in one step) """
        if message_from_central_controller == 'request_availability':
            # leave power as it is and responde availability
            message_to_central_controller = self._availability_data[i] # if the controller / the flexibility is available, 1, else 0
            control_power = self._control_power
        elif message_from_central_controller == 'done':
            # leave power as is and responde something (it is not used anyways in the central controller)
            message_to_central_controller = 'Thank you!'
            control_power = self._control_power
        elif self._availability_data[i]:
            # In this case, the central controller should send a numeric power value
            message_to_central_controller = 'power is set!'
            self._control_power = float(message_from_central_controller)
            control_power = self._control_power
        else:
            # depending on how the simulation is set up, this case should not happen. So just continue with the same power value
            message_to_central_controller = 'nothing happening here'
            control_power = self._control_power

        return message_to_central_controller, -control_power


