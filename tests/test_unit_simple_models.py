import unittest
import sys
from models import simple_test_models as models            # modul under test


class TestSimpleModels(unittest.TestCase):
    def test_SimpleTestPVModel(self):
        model = models.SimpleTestPVModel()
        self.assertEqual(model.step(0), -2)

    def test_SimpleTestHouseholdModel(self):
        model = models.SimpleTestHouseholdModel()
        self.assertEqual(model.step(2), 2)

    def test_SimpleTestGridModel(self):
        model = models.SimpleTestGridModel()
        self.assertEqual(model.step(0, [2,-1,6]), 7)

    def test_SimpleTestFlexibilityModel(self):
        model = models.SimpleTestFlexibilityModel()
        consumed_power, capacity = model.step(0, control_input=2) # normal charging
        self.assertEqual(consumed_power, 2, 'normal charging failed')
        self.assertEqual(capacity, 2, 'normal charging failed')
        consumed_power, capacity = model.step(1, control_input=-1) # normal discharge
        self.assertEqual(consumed_power, -1, 'normal discharge failed')
        self.assertEqual(capacity, 1, 'normal discharge failed')
        consumed_power, capacity = model.step(2, control_input=-2) # try to discharge below zero
        self.assertEqual(consumed_power, -1, 'discharging below zero failed')
        self.assertEqual(capacity, 0, 'discharging below zero failed')
        consumed_power, capacity = model.step(3, control_input=6) # try to charge above maximum
        self.assertEqual(consumed_power, 5, 'charging above maximum failed')
        self.assertEqual(capacity, 5, 'charging above maximum failed')

    def test_SimpleTestFlexibilityModelWithAvailabilityModel(self):
        model = models.SimpleTestFlexibilityWithAvailabilityModel()
        for i, [contr_signal, required_capacity, available] in enumerate(zip(
            [1, 1, 1, -1, -1, -1, 1, 1, 1, 1],
            [1, 2, 3,  2,  1,  0, 1, 1, 1, 1],
            model._availability)):
            consumed_power, capacity = model.step(i, contr_signal)
            self.assertEqual(consumed_power, contr_signal*available, 'availability failed')
            self.assertEqual(capacity, required_capacity, 'availability failed')

    def test_SimpleTestCentralFlexibilityControllerModel(self):
        model = models.SimpleTestCentralFlexibilityControllerModel()
        messages          = [['nothing happening here', 'power is set!'], [1, 1], ['Thank you!', 'Thank you!']]
        expected_responses = [                     'request_availability',    1.0,                      'done']
        for i, [message, exp_resp] in enumerate(zip(messages, expected_responses)):
            proceede, response = model.step(0, power_difference=2, messages_from_controllers=message)
            self.assertEqual(exp_resp, response, 'Unexpected response from central controller')

    def test_SimpleTestDecentralFlexibilityControllerModel(self):
        model = models.SimpleTestDecentralFlexibilityControllerModel()
        messages                         = ['request_availability',                      1.0,       'done']        
        expected_responses_not_available = [                     0, 'nothing happening here', 'Thank you!']
        expected_responses_available     = [                     1,          'power is set!', 'Thank you!']
        expected_ctr_pwr_not_available   = [                     0,                        0,            0]
        expected_ctr_pwr_available       = [                     0,                     -1.0,         -1.0]
        
        # step i not available
        i = 7
        for j, [message, exp_resp, expected_ctr_pwr] in enumerate(zip(messages, expected_responses_not_available, expected_ctr_pwr_not_available)):
            response, control_power = model.step(i, message_from_central_controller=message)
            self.assertEqual(exp_resp, response, f'Unexpected response from Decentral controller for request "{message}"')
            self.assertEqual(expected_ctr_pwr, control_power, f'Unexpected control power for requst "{message}"')
        
        # step 6 available
        i = 0
        for j, [message, exp_resp, expected_ctr_pwr] in enumerate(zip(messages, expected_responses_available, expected_ctr_pwr_available)):
            response, control_power = model.step(i, message_from_central_controller=message)
            self.assertEqual(exp_resp, response, f'Unexpected response from Decentral controller for request "{message}"')
            self.assertEqual(expected_ctr_pwr, control_power, f'Unexpected control power for requst "{message}"')
    

if __name__ == '__main__':
    unittest.main()

