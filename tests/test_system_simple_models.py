import unittest
import sys
from models import simple_test_models as models             # modul under test

class SystemTestSimpleModels(unittest.TestCase):    
    def test_SystemSimpleModels(self):
        pv_model = models.SimpleTestPVModel()
        household_model = models.SimpleTestHouseholdModel()
        grid_model = models.SimpleTestGridModel()
        flex_model = models.SimpleTestFlexibilityWithAvailabilityModel()
        
        central_controller = models.SimpleTestCentralFlexibilityControllerModel()
        decentral_controller = models.SimpleTestDecentralFlexibilityControllerModel()

        exp_pv_power          = [-2.0, -2.0, -2.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0]
        exp_household_power   = [ 2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  4.0,  0.0,  4.0]
        exp_flexibility_power = [-0.0, -0.0, -0.0, -0.0,  2.0, -0.0,  2.0,  0.0,  0.0,  0.0]
        exp_grid_import_power = [ 0.0,  0.0,  0.0, -2.0,  0.0, -2.0,  0.0,  0.0, -4.0,  0.0] 
        
        grid_import_power = 0
        for i, exp_output in enumerate(zip(exp_pv_power, exp_household_power, exp_flexibility_power, exp_grid_import_power)):
            pv_power = pv_model.step(i)
            household_power = household_model.step(i)

            proceede = False
            decentral_controller_message = ''
            while not proceede:
                proceede, central_controller_message = central_controller.step(
                    i, grid_import_power, [decentral_controller_message])
                decentral_controller_message, control_signal = decentral_controller.step(
                    i, central_controller_message)
            
            flexibility_power, _ = flex_model.step(i, control_signal)
            grid_import_power = grid_model.step(i, [pv_power, household_power, flexibility_power])

            # print(f'{pv_power=:2.1f}, {household_power=:2.1f}, {flexibility_power=:2.1f}, {grid_import_power=:2.1f}')
            self.assertEqual([pv_power, household_power, flexibility_power, grid_import_power], list(exp_output), 'One of the outputs is behaving unexpectedly')

                


if __name__ == '__main__':
    unittest.main()