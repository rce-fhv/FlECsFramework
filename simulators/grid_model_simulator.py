# simulator_mosaik.py
"""
Mosaik model of the grid (implemented direcly in the simulator).

"""
import mosaik_api_v3
import pandas as pd
import pathlib

META = {
    'type': 'time-based',
    'models': {
        'GridModel': {
            'public': True,
            'params': ['clockfun'],
            'attrs': ['node_power', 'residual_power'],
        },
    },
}


class GridSim(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'Grid_'
        self.time = -1
        self.residual_power = 0

        self.output = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['model', 'attribute']))

    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        self.output_path = output_dir.joinpath(self.eid_prefix)
        self.time_resolution = time_resolution
        return self.meta
    
    def create(self, num, model, clockfun):
        '''
        Simple Grid model, which sums up all its provided inputs and calculates the residual power (from and to the transformer)
        loads negative, feed in positive
        '''
        self.clockfun = clockfun

        if num > 1: 
            raise ValueError('Can only create one Grid Model')
        
        entities = [{'eid': 'Grid_1', 'type': model}]

        return entities

    def step(self, time, inputs, max_advance):
        assert time - self.time == 1, 'Something is off with the timing in the Grid'
        self.time = time
        datetime = self.clockfun(time)

        assert len(inputs) <= 1 , 'Only one grid should exist!'
        attrs = inputs['Grid_1']
        
        # print(f'grid simulator here: attrs={attrs["node_power"]}')

        self.residual_power = sum(attrs['node_power'].values())

        #save the outputs
        self.output.loc[datetime, ('grid_0', 'residual power')] = self.residual_power

        return time + 1  # Step in smallest time resolution (step size is specified in self.time_resolution)

    def get_data(self, outputs):
        assert len(outputs) == 1, 'Only one grid exists, no other outputs possible' 
        assert len(outputs['Grid_1']) == 1, 'Grid only has one attributte: residual_power'
         
        return {'Grid_1': {'residual_power': self.residual_power}}
    
    def finalize(self):
        # TODO Improve paths!
        self.output.to_pickle(self.output_path.with_suffix('.pkl'))


def main():
    return mosaik_api_v3.start_simulation(GridSim())


if __name__ == '__main__':
    main()