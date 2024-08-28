# simulator_mosaik.py
"""
Mosaik interface for the battery storage model.

"""
import mosaik_api_v3
from models.battery_storage_models import StorageModelWithEfficiency, StorageModelWithSelfdischarge
import pandas as pd
import pathlib

META = {
    'type': 'time-based',
    'models': {
        'StorageModel': {
            'public': True,
            'params': ['clockfun', 'capacity_0', 'battery_params'],
            'attrs': ['power', 'actual_power', 'capacity'],
        },
        'StorageModelWithSelfdischarge': {
            'public': True,
            'params': ['clockfun', 'capacity_0', 'battery_params'],
            'attrs': ['power', 'actual_power', 'capacity'],
        }
    }
}

storage_models = {'StorageModel': StorageModelWithEfficiency,
                  'StorageModelWithSelfdischarge': StorageModelWithSelfdischarge
                  }

class StorageModelSim(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'StorageModel_'
        self.entities = {}  # Maps EIDs to model instances/entities
        self.time = -1

        self.output = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['model', 'attribute']))
    
    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        self.output_path = output_dir.joinpath(self.eid_prefix)
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, clockfun, capacity_0, battery_params):
        next_eid = len(self.entities)
        entities = []
        self.clockfun = clockfun

        for i in range(next_eid, next_eid + num):
            model_instance = storage_models[model](capacity_0, delta_t=self.time_resolution, **battery_params) # instantiate model from global dict
            eid = f'{self.eid_prefix}{i}'
            self.entities[eid] = model_instance

            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        assert time - self.time == 1, 'Something is off with the timing in the StorageModel'
        self.time = time
        datetime = self.clockfun(time)

        for eid, model_instance in self.entities.items():
            attrs = inputs[eid]
            power = attrs['power']
            assert len(power) == 1, 'Only one input allowed for storage power'
            model_instance.step(sum(power.values()), time)

            #save the outputs
            self.output.loc[datetime, (eid, 'power')] = model_instance.power
            self.output.loc[datetime, (eid, 'capacity')] = model_instance.capacity
            self.output.loc[datetime, (eid, 'actual_power')]  = model_instance.actual_power
            
        # TODO: time resolution should be flexible
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
        # TODO Improve paths!
        self.output.to_pickle(self.output_path.with_suffix('.pkl'))


def main():
    return mosaik_api_v3.start_simulation(StorageModelSim())


if __name__ == '__main__':
    main()