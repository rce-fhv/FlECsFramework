# simulator_mosaik.py
"""
Mosaik interface for the Distribution System Operator (DSO) model, managing the smart meter data.

"""
import mosaik_api_v3
import pandas as pd
from models.DSO_model import DSOModel
import pathlib


META = {
    'type': 'time-based',
    'models': {
        'DSOModel': {
            'public': True,
            'params': ['clockfun'],
            'attrs': ['node_power', 'node_data', 'ec_data'],
        },
    },
}


class DSOModelSim(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'DSOModel_'
        self.entities = {}  # Maps EIDs to model instances/entities
        self.time = -1

        # self.output = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['model', 'attribute']))

    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        self.output_dir = output_dir # here differnt to the others, as there are more than one file to save
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, clockfun):
        next_eid = len(self.entities)
        entities = []
        self.clockfun = clockfun

        for i in range(next_eid, next_eid + num):
            model_instance = DSOModel(clockfun=self.clockfun, time_resolution=self.time_resolution)
            eid = f'{self.eid_prefix}{i}'
            self.entities[eid] = model_instance

            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        assert time - self.time == 1, 'Something is off with the timing in the DSO Model'
        self.time = time

        for eid, model_instance in self.entities.items():
            attrs = inputs[eid]
            model_instance.step(time, attrs['node_power'])
        return time + 1  # Step in smallest time resolution (step size is specified in self.time_resolution)
    

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            model = self.entities[eid]
            data[eid] = {}
            for attr in attrs:
                data[eid][attr] = getattr(model, attr)

        return data
    
    def finalize(self):
        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        for eid, model_instance in self.entities.items():
            model_instance._node_data.to_pickle(output_dir.joinpath(f'{eid}_node_data.pkl'))
            model_instance._electricity_ec.to_pickle(output_dir.joinpath(f'{eid}_electricity_ec.pkl'))
            model_instance._electricity_supplier.to_pickle(output_dir.joinpath(f'{eid}_electricity_supplier.pkl'))
            
            