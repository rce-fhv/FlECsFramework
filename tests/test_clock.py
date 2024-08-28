import unittest
import sys
import pandas as pd
from models.clock import clock

class TestClockModel(unittest.TestCase):
    def test_clock(self):
        clockfun1 = clock(15*60)
        clockfun2 = clock(60*60, origin_timestamp='2024-01-01 00:00:00')
        self.assertEqual(clockfun1(7), pd.Timestamp('2022-01-01 01:45:00+0100', tz='Europe/Vienna'))
        self.assertEqual(clockfun2(7), pd.Timestamp('2024-01-01 07:00:00+0100', tz='Europe/Vienna'))
        self.assertEqual(clockfun2(8), pd.Timestamp('2024-01-01 08:00:00+0100', tz='Europe/Vienna'))