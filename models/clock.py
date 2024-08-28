import pandas as pd


def clock(time_resolution, origin_timestamp='2022-01-01 00:00:00', tz='Europe/Vienna', unit='s'):
    '''
    Returns a clock function
    time_resolution: seconds (or units) per integer time step
    origin_timestamp: start time of the clock in UTC (defaults to '2022-01-01 00:00:00')
    timezone: time zone as string as specified by pandas, pass None if timezone unaware
    unit: unit of the time resolution (default = s)
    '''
    def clockfun(int_time):
        '''
        Convert from integer time stamps to datetime
        '''
        return pd.to_datetime(int_time * time_resolution, unit=unit, origin=pd.Timestamp(origin_timestamp)).tz_localize('UTC').tz_convert(tz)
    
    return clockfun