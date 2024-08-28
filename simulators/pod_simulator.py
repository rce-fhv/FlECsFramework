# simulator_mosaik.py
"""
Mosaik model of a Point of Delivery (implemented direcly in the simulator).
Gives the possibilty, to connect several loads and or producers to the grid and the dso as one single entity.
btm: Behind the Meter
grid: power to and from the grid
"""

import mosaik_api_v3
import pandas as pd
import numpy as np
import pathlib

META = {
    'type': 'time-based',
    'models': {
        'PODModel': {
            'public': True,
            'params': ['clockfun'],
            'attrs': ['btm_power', 'grid_power'],
        },
    },
}


class PODSim(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'POD_'
        self.time = -1 
        self.entities_grid_power = {}  # Maps EIDs to the current grid_power

        self.output = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['model', 'attribute']))

    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        self.output_path = output_dir.joinpath(self.eid_prefix)
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, clockfun):
        self.clockfun = clockfun
        next_eid = len(self.entities_grid_power)
        entities = []

        for i in range(next_eid, next_eid + num):
            eid = '%s%d' % (self.eid_prefix, i)
            self.entities_grid_power[eid] = 0
            entities.append({'eid': eid, 'type': model})

        return entities
    
    def step(self, time, inputs, max_advance):
        self.time = time
        for eid, grid_power in self.entities_grid_power.items():
            if eid in inputs:
                btm_powers = inputs[eid]['btm_power']
                grid_power = sum(btm_powers.values())
            else:
                grid_power = np.nan

            self.entities_grid_power[eid] = grid_power

            #save the outputs
            self.output.loc[self.clockfun(time), (eid, 'grid_power')] = grid_power

        return time + 1  # Step size is 1 second

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            for attr in attrs:
                assert attr in self.meta['models']['PODModel']['attrs'], 'Unknown output attribute: %s' % attr
            data['time'] = self.time
            data[eid] = {'grid_power': self.entities_grid_power[eid]}

        return data
    
    def finalize(self):
        # TODO Improve paths!
        self.output.to_pickle(self.output_path.with_suffix('.pkl'))

