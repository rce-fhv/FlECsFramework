import unittest
import sys
import pandas as pd
from models.load_model import LoadModel


class TestLoadModel(unittest.TestCase):
    def setUp(self):
        clockfun = lambda int_time: pd.to_datetime(int_time * 60*60, unit='s', origin=pd.Timestamp('2010-01-01 00:00:00')).tz_localize('Europe/Vienna')
        self.load_model = LoadModel(clockfun)

    def test_initialization(self):
        self.assertEqual(type(self.load_model.data), pd.DataFrame)

    def test_step(self):
        self.assertAlmostEqual(self.load_model.step(0)[0], 526.1833333333333) # Sample test first model output 
        self.assertAlmostEqual(self.load_model.step(0)[35], 1174.5833333333333) # Sample test last model output