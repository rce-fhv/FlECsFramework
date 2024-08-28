import matplotlib as mpl
from pandas._libs.tslibs.parsing import guess_datetime_format
import re


def plot_time_discrete(ax, x, y, *args, **kwargs):
    '''
    Plots either a step function in between a time index if len(x) == len(y)+1 or a continuous plot when len(x) == len(y)
    x: time values of the plot
    y: corersponding y values in between two x values
    
    visualization...
    either plot step plot
        y1        y2
    x1--------x2--------x3

    or continuous plot
    y1        y2        y3
    x1--------x2--------x3
    '''
    if len(x) == len(y) +1:
        # where is specified here and can lead to wrong behaviour
        kwargs.pop('where', '')

        # extend the y values
        y = list(y)
        y = y + [y[-1]]
        ax.step(x, y, where='post', *args, **kwargs)
    elif len(x) == len(y):
        ax.plot(x, y, *args, **kwargs)
    else:
        raise ValueError('Length of x any y does not match, it can either be len(x) == len(y) or len(x) == len(y)+1')

# mpl.axes.Axes.plot_time_discrete = plot_time_discrete


class TimeMatcher:
    def __init__(self, timestring) -> None:
        '''
        Create a timestamp from a string with '*' as placeholders for matching with other string
        '''
        self.regex = timestring.replace('.', '\.').replace('*', '\d')  
        self.dt_format = guess_datetime_format(timestring.replace('****', '2000').replace('*', '1'))
        if self.dt_format == None:
            raise ValueError('timestring could not be identified properly')

    def __eq__(self, __value: object) -> bool:
        strf_timestamp = __value.strftime(self.dt_format)
        return re.match(self.regex, strf_timestamp) != None