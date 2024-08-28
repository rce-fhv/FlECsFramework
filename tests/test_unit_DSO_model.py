import unittest
import sys
import pandas as pd
sys.path.append('./FlECs_Frameworks/') 
from models.DSO_model import DSOModel


class TestDSOModel(unittest.TestCase):
    def setUp(self):
        clockfun = lambda int_time: pd.to_datetime(int_time * 60*15, unit='s', origin=pd.Timestamp('2022-01-01 00:00:00'))
        self.dso_model = DSOModel(clockfun)

    def test_calc_share_dynamically(self):
        self.dso_model.step(1, {'h0':-2, 'h1':-2, 'h3':4, 'h4':5})
        self.dso_model.step(2, {'h0':-2, 'h1':-10, 'h3':4, 'h4':5})
        self.dso_model.step(3, {'h0':-2, 'h1':5, 'h3':4, 'h4':5})
        self.dso_model.calc_share_dynamically()
        self.dso_model.step(4, {'h0':-2, 'h1':5, 'h3':4, 'h4':5})
        self.dso_model.calc_share_dynamically()

        for v in list(self.dso_model.electricity_ec.sum(axis=1)):
                    self.assertAlmostEqual(v, 0.)