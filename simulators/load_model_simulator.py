# simulator_mosaik.py
"""
Mosaik interface for the Load model.

"""
import mosaik_api_v3
from models.load_model import LoadModel, LOADDEFAULTDATANAME
import pandas as pd
import pathlib


META = {
    'type': 'time-based',
    'models': {
        'LoadModel': {
            'public': True,
            'params': ['clockfun', 'data_name'],
            'attrs': ['load'],
        },
    },
}
        

class LoadModelSim(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'LoadModel_'
        self.entities_lookup = {} # Maps eid to dataframe columns of load model
        self.load_time_resolution = 3600 # s

        self.output = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['model', 'attribute']))

        self.data = {}

    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        self.output_path = output_dir.joinpath(self.eid_prefix)
        self.time_resolution = time_resolution
        assert self.load_time_resolution % time_resolution == 0, 'Load Model expects time resosultion to be an integer fraction of its own time resolution'
        self.int_steps_size = int(self.load_time_resolution // time_resolution)
        return self.meta

    def create(self, num, model, clockfun, data_name=LOADDEFAULTDATANAME):
        self.model = LoadModel(clockfun, data_name, num)

        next_eid = len(self.entities_lookup)
        entities = []

        for i in range(next_eid, next_eid + num):            
            eid = f'{self.eid_prefix}{i}'
            self.entities_lookup[eid] = self.model.data.columns[i]  # Maps eid to dataframe columns of load model
            
            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        self.time = time

        assert len(inputs) == 0, 'Load Model expects no inputs'

        datetime, model_data = self.model.step(time)

        for eid, df_index in self.entities_lookup.items():
            out = model_data[df_index]
            self.data[eid] = {'load': out}
            
            # save the outputs
            self.output.loc[datetime, (eid, 'load')] = out

        return time + self.int_steps_size  # Step in smallest time resolution (step size is specified in self.time_resolution)

    def get_data(self, outputs):        
        # data = {}
        # for eid, attrs in outputs.items():
        #     data[eid] = {}
        #     for attr in attrs:
        #         data[eid][attr] = self.data[eid][attr]

        return {eid: {attr: self.data[eid][attr] for attr in attrs} for eid, attrs in outputs.items()}
    
    def finalize(self):
        # TODO Improve paths!
        self.output.to_pickle(self.output_path.with_suffix('.pkl'))


# def main():
#     return mosaik_api_v3.start_simulation(PVModelSim())

if __name__ == '__main__':
    pass
    # main()