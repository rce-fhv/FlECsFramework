import unittest
import sys
import pandas as pd
from utility.utility import TimeMatcher

class TestTimeMatcher(unittest.TestCase):
    def test_time_matcher1(self):
        self.assertTrue(pd.Timestamp('2022-12-01 22:15:00') == TimeMatcher('****-**-01 22:15:00'))

    def test_time_matcher2(self):
        self.assertFalse(pd.Timestamp('2022-12-02 22:15:00') == TimeMatcher('****-**-01 22:15:00'))
    
    def test_time_matcher3(self):
        self.assertTrue(pd.Timestamp('2022-12-01 22:15:00.20') == TimeMatcher('****-**-01'))

    def test_time_matcher4(self):
        self.assertTrue(pd.Timestamp('2022-12-01 22:15:00.20') == TimeMatcher('****-**-01 22:15:00.20'))

    # def test_time_matcher5(self):
    # not yet implemented
    #     self.assertTrue(pd.Timestamp('2022-12-01 22:15:00.20') == TimeMatcher('22:15:00.20'))
        