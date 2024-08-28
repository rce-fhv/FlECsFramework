import unittest
import sys
import pandas as pd
from models.PV_model import PV, PVperfectPrediction

class TestPVModel(unittest.TestCase):
    def setUp(self):
        clockfun = lambda int_time: pd.to_datetime(int_time * 60*60, unit='s', origin=pd.Timestamp('2010-01-01 00:00:00')).tz_localize('Europe/Vienna')
        self.pv = PV(clockfun, download_data=True, data_name='test_data', params_list=[{'peakpower':5}, {'peakpower':10}, {'peakpower':15}, {'peakpower':20}])

    def test_default_initialization(self):
        self.assertEqual(self.pv.data.index[0].year, 2009)  # time index has to start one year before the relevant year due to time zone conversion
        self.assertEqual(self.pv.data.index[-1].year, 2011) # time index goes until '2011-01-01 00:00:00+01:00'

    def test_step(self):
        self.assertEqual(self.pv.step(12), {0: 2458.45, 1: 4916.9, 2: 7375.35, 3: 9833.8}, 'PV output with default initialization different as expected')

class TestPVModelPerfectPrediction(unittest.TestCase):
    def setUp(self):
        clockfun = lambda int_time: pd.to_datetime(int_time * 60*60, unit='s', origin=pd.Timestamp('2010-01-01 00:00:00')).tz_localize('Europe/Vienna')
        # make sure data is downloaded
        self.pv = PV(clockfun, download_data=True, data_name='test_data', params_list=[{'peakpower':5}, {'peakpower':10}, {'peakpower':15}, {'peakpower':20}])

        self.pv_perfect_prediction = PVperfectPrediction(clockfun, 'test_data', 5)

    def test_step(self):
        pass
        # print(self.pv_perfect_prediction.step(12))
        # df = pd.DataFrame.from_dict({
        #     '2010-01-01 12:00:00+01:00':    24584.5,
        #     '2010-01-01 13:00:00+01:00':    20576.0,
        #     '2010-01-01 14:00:00+01:00':    23326.0,
        #     '2010-01-01 15:00:00+01:00':     4793.0,
        #     '2010-01-01 16:00:00+01:00':      183.5,
        #     '2010-01-01 17:00:00+01:00':        0.0}, orient='index')
        # df.index = pd.to_datetime(df.index)
        # self.assertTrue(self.pv_perfect_prediction.step(12).equals(df), 'PV output with default initialization different as expected')
        # TODO Fix Test!