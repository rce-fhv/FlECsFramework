import unittest
import pandas as pd
import numpy as np
from models.controller.MPC_battery_storage_controller import MILPBatteryModel, MPCBatteryStorageController  # ResidualPowerPrediction,


class TestMILPBatteryModel(unittest.TestCase):
    def setUp(self) -> None:
        self.milp_model = MILPBatteryModel()

    def test_initialization(self):
        self.assertEqual(self.milp_model.delta_t             ,        3600) # s
        self.assertEqual(self.milp_model.max_capacity        , 5000 * 3600) # J
        self.assertEqual(self.milp_model.min_capacity        ,           0) # J
        self.assertEqual(self.milp_model.eta_i               ,        0.95) #
        self.assertEqual(self.milp_model.eta_o               ,        0.95) #
        self.assertEqual(self.milp_model.max_power_charge    ,        2000) # W
        self.assertEqual(self.milp_model.max_power_discharge ,        2000) # W

    def test_solve(self):
        power_output = self.milp_model.solve(capacity_0=500*3600, power_storage_0=-1000, residual_power_prediction=np.array([1000, 1000]), make_full_output=False)
        self.assertListEqual(list(power_output), [-1000, -1000])
    
    def test_solve_longer(self):
        power_output = self.milp_model.solve(capacity_0=500*3600, power_storage_0=-1000, residual_power_prediction=np.array([1000, 1000, -100, 100]), make_full_output=False)
        self.assertListEqual(list(power_output), [-1000, -1000, 100, -100])

    def test_solve_powerlimit_upper(self):
        power_output = self.milp_model.solve(capacity_0=500*3600, power_storage_0=-1000, residual_power_prediction=np.array([1000, 3000]), make_full_output=False)
        self.assertListEqual(list(power_output), [-1000, -2000])

    def test_solve_powerlimit_lower(self):
        power_output = self.milp_model.solve(capacity_0=2500*3600, power_storage_0=0, residual_power_prediction=np.array([0, -3000]), make_full_output=False)
        self.assertListEqual(list(power_output), [0, 2000])

    def test_solve_capycity_limit_upper(self):
        power_output = self.milp_model.solve(capacity_0=3000*3600, power_storage_0=0, residual_power_prediction=np.array([0, 1100, 1100]), make_full_output=False)
        self.assertAlmostEqual(sum(power_output), -2000/0.95)

    def test_solve_capycity_limit_lower(self):
        power_output = self.milp_model.solve(capacity_0=2000*3600, power_storage_0=0, residual_power_prediction=np.array([0, -1100, -1100]), make_full_output=False)
        self.assertAlmostEqual(sum(power_output), 2000*0.95)


# class TestResidualPowerPrediction(unittest.TestCase):
#     def setUp(self) -> None:
#         self.power_prediction = ResidualPowerPrediction()

#     def test_initial_set_perfect_prediction(self):
#         test_df = pd.DataFrame.from_dict({0: -1000, 1: 0, 2: 0, 3: 0, 4: 10000, 5: -1000, 6: 1000, 7: 1000, 8: 100}, orient='index')
#         self.power_prediction.set_perfect_prediction(test_df)
#         self.assertTrue(test_df.equals(self.power_prediction.residual_power_prediction_df))

#     def test_set_perfect_prediction(self):
#         # update prediction once
#         self.power_prediction.set_perfect_prediction(
#             pd.DataFrame.from_dict({0: -1000, 1: 0, 2: 0, 3: 0}, orient='index'))
#         # update prediction twice
#         self.power_prediction.set_perfect_prediction(
#             pd.DataFrame.from_dict({4: 10000, 5: -1000, 6: 1000, 7: 1000, 8: 100}, orient='index'))
        
#         self.assertTrue(
#             pd.DataFrame.from_dict({0: -1000, 1: 0, 2: 0, 3: 0, 4: 10000, 5: -1000, 6: 1000, 7: 1000, 8: 100}, orient='index')
#             .equals(self.power_prediction.residual_power_prediction_df))
        
#     def test_set_perfect_prediction_datetime(self):
#         # update prediction once
#         test_df1 = pd.DataFrame.from_dict({'2022-01-01 00:00:00': -1000, '2022-01-01 00:15:00': 0, '2022-01-01 00:30:00': 0, '2022-01-01 00:45:00': 0}, orient='index')
#         test_df1.index = pd.to_datetime(test_df1.index)
#         self.power_prediction.set_perfect_prediction(test_df1)

#         # update prediction twice
#         test_df2 = pd.DataFrame.from_dict({'2022-01-01 01:00:00': 10000, '2022-01-01 01:15:00': -1000, '2022-01-01 01:30:00': 1000, '2022-01-01 01:45:00': 1000, '2022-01-01 02:00:00': 100}, orient='index')
#         test_df2.index = pd.to_datetime(test_df2.index)
#         self.power_prediction.set_perfect_prediction(test_df2)

#         test_df = pd.DataFrame.from_dict({'2022-01-01 00:00:00': -1000, '2022-01-01 00:15:00': 0, '2022-01-01 00:30:00': 0, '2022-01-01 00:45:00': 0, '2022-01-01 01:00:00': 10000, '2022-01-01 01:15:00': -1000, '2022-01-01 01:30:00': 1000, '2022-01-01 01:45:00': 1000, '2022-01-01 02:00:00': 100}, orient='index')
#         test_df.index = pd.to_datetime(test_df.index)
        
#         self.assertTrue(test_df.equals(self.power_prediction.residual_power_prediction_df))
        
#     def test_set_perfect_prediction_w_overlap(self):
#         # update prediction once
#         self.power_prediction.set_perfect_prediction(
#             pd.DataFrame.from_dict({0: -1000, 1: 0, 2: 0, 3: 0}, orient='index'))
#         # update prediction twice
#         self.power_prediction.set_perfect_prediction(
#             pd.DataFrame.from_dict({3: 0, 4: 10000, 5: -1000, 6: 1000, 7: 1000, 8: 100}, orient='index'))
        
#         self.assertTrue(
#             pd.DataFrame.from_dict({0: -1000, 1: 0, 2: 0, 3: 0, 4: 10000, 5: -1000, 6: 1000, 7: 1000, 8: 100}, orient='index')
#             .equals(self.power_prediction.residual_power_prediction_df))
        
#     def test_set_perfect_prediction_w_overlap_datetime(self):
#         # update prediction once
#         test_df1 = pd.DataFrame.from_dict({'2022-01-01 00:00:00': -1000, '2022-01-01 00:15:00': 0, '2022-01-01 00:30:00': 0, '2022-01-01 00:45:00': 0, '2022-01-01 01:00:00': 10000, '2022-01-01 01:15:00': -1000}, orient='index')
#         test_df1.index = pd.to_datetime(test_df1.index)
#         self.power_prediction.set_perfect_prediction(test_df1)

#         # update prediction twice
#         test_df2 = pd.DataFrame.from_dict({'2022-01-01 01:00:00': 10000, '2022-01-01 01:15:00': -1000, '2022-01-01 01:30:00': 1000, '2022-01-01 01:45:00': 1000, '2022-01-01 02:00:00': 100}, orient='index')
#         test_df2.index = pd.to_datetime(test_df2.index)
#         self.power_prediction.set_perfect_prediction(test_df2)

#         test_df = pd.DataFrame.from_dict({'2022-01-01 00:00:00': -1000, '2022-01-01 00:15:00': 0, '2022-01-01 00:30:00': 0, '2022-01-01 00:45:00': 0, '2022-01-01 01:00:00': 10000, '2022-01-01 01:15:00': -1000, '2022-01-01 01:30:00': 1000, '2022-01-01 01:45:00': 1000, '2022-01-01 02:00:00': 100}, orient='index')
#         test_df.index = pd.to_datetime(test_df.index)
        
#         self.assertTrue(test_df.equals(self.power_prediction.residual_power_prediction_df))
        
#     def test_get_prediction(self):
#         test_df = pd.DataFrame.from_dict({0: -1000, 1: 0, 2: 0, 3: 0, 4: 10000, 5: -1000, 6: 1000, 7: 1000, 8: 100}, orient='index')
#         self.power_prediction.set_perfect_prediction(test_df)
#         pred = self.power_prediction.get_current_prediction(4)
#         self.assertSequenceEqual(list(pred), list(np.array([10000,-1000, 1000, 1000, 100])))

#     def test_get_prediction_datetime(self):
#         test_df = pd.DataFrame.from_dict({'2022-01-01 00:00:00': -1000, '2022-01-01 00:15:00': 0, '2022-01-01 00:30:00': 0, '2022-01-01 00:45:00': 0, '2022-01-01 01:00:00': 10000, '2022-01-01 01:15:00': -1000, '2022-01-01 01:30:00': 1000, '2022-01-01 01:45:00': 1000, '2022-01-01 02:00:00': 100}, orient='index')
#         test_df.index = pd.to_datetime(test_df.index)
#         self.power_prediction.set_perfect_prediction(test_df)
#         pred = self.power_prediction.get_current_prediction('2022-01-01 01:00:00')
#         self.assertSequenceEqual(list(pred), list(np.array([10000,-1000, 1000, 1000, 100])))

#     def test_get_prediction_type(self):
#         test_df = pd.DataFrame.from_dict({0: -1000, 1: 0, 2: 0, 3: 0, 4: 10000, 5: -1000, 6: 1000, 7: 1000, 8: 100}, orient='index')
#         self.power_prediction.set_perfect_prediction(test_df)
#         pred = self.power_prediction.get_current_prediction(4)
#         self.assertIsInstance(pred, np.ndarray)


class TestMPCBatteryStorageController(unittest.TestCase):
    def setUp(self):
        # Mock the clock function
        clockfun = lambda int_time: pd.to_datetime(int_time * 60*15, unit='s', origin=pd.Timestamp('2022-01-01 00:00:00'))
        # Initialize the controller
        self.controller = MPCBatteryStorageController(clockfun)

    def test_step_1(self):
        prediction_data = pd.DataFrame.from_dict({'2022-01-01 00:00:00': 0, '2022-01-01 00:15:00': -1000}, orient='index')
        prediction_data.index = pd.to_datetime(prediction_data.index)
        self.controller.update_perfect_prediction(prediction_data)
        power = self.controller.step(0, 2000*3600)
        self.assertAlmostEqual(power, 1000)

    def test_step_2(self):
        prediction_data = pd.DataFrame.from_dict({'2022-01-01 00:00:00': 0, '2022-01-01 00:15:00': 1000}, orient='index')
        prediction_data.index = pd.to_datetime(prediction_data.index)
        self.controller.update_perfect_prediction(prediction_data)
        power = self.controller.step(0, 2000*3600)
        self.assertAlmostEqual(power, -1000)

