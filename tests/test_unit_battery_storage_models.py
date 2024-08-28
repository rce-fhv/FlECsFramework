import unittest
import sys
from models.battery_storage_models import StorageModelWithEfficiency


class TestStorageModelWithEfficiency(unittest.TestCase):

    def setUp(self):
        # Initialize a BatteryStorageModel instance for testing
        self.battery = StorageModelWithEfficiency()

    def test_initial_conditions(self):
        # Test if the battery is correctly initialized
        self.assertEqual(self.battery.delta_t            ,        3600)  # s
        self.assertEqual(self.battery.capacity           , 2500 * 3600)  # J
        self.assertEqual(self.battery.max_capacity       , 5000 * 3600)  # J 
        self.assertEqual(self.battery.min_capacity       ,           0)  # J 
        self.assertEqual(self.battery.eta_i              ,        0.95)
        self.assertEqual(self.battery.eta_o              ,        0.95)
        self.assertEqual(self.battery.max_power_charge   ,        2000)  # W
        self.assertEqual(self.battery.max_power_discharge,        2000)  # W

    def test_step_function_discharge(self):
        # Test the step function during discharge
        actual_power, capacity = self.battery.step(-500)  # Discharging with 500 W
        self.assertEqual(actual_power, -500)

    def test_step_function_charge(self):
        # Test the step function during charge
        actual_power, capacity = self.battery.step(500)  # Charging with 500 W
        self.assertEqual(actual_power, 500)

    def test_over_discharge_protection(self):
        # Test over-discharge protection
        for _ in range(1000):  # Discharge for a long time
            actual_power, capacity = self.battery.step(-1000) # Discharge with 1000 W
        self.assertGreaterEqual(capacity, 0)  # capycity should not go below 0
        self.assertEqual(actual_power, 0)  # actual power should now be zero

    def test_over_charge_protection(self):
        # Test over-charge protection
        for _ in range(1000):  # Discharge for a long time
            actual_power, capacity = self.battery.step(1000) # Discharge with 1000 W
        self.assertGreaterEqual(capacity, 5000 * 3600)  # capacity should not go above max capacity
        self.assertEqual(actual_power, 0)  # actual power should now be zero

    def test_maximum_power_protection(self):
        # Test maximum_power protection
        actual_power, capacity = self.battery.step(10000)  # charge with 10000 W
        self.assertEqual(actual_power, 2000) # Power schould be 2000 W

    def test_minimum_power_protection(self):
        # Test minimum_power protection
        actual_power, capacity = self.battery.step(-10000)  # discharge with 10000 W
        self.assertEqual(actual_power, -2000) # Power schould be -2000 W

    def test_minimum_power_protection_and_overdischarge(self):
        # Test minimum_power protection
        actual_power, capacity = self.battery.step(-10000)  # discharge with 1500 W
        # actual_power, capacity = self.battery.step(-10000)  # discharge with 1500 W
        self.assertEqual(actual_power, -2000) # Power schould be -2000 W