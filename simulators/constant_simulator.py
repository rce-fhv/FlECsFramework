# simulator_mosaik.py
"""
Mosaik simulator with constant output.

"""
import mosaik_api_v3
import pathlib

META = {
    'type': 'time-based',
    'models': {
        'ConstantSimulator': {
            'public': True,
            'params': ['value'],
            'attrs': ['value'],
        },
    },
}


class ConstantSimulator(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'Constant_'
        self.entities = {}  # Maps EIDs to models constant value: {EID: value, ...}

    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        self.output_dataframe_filename = output_dir.joinpath(self.eid_prefix.replace('_', '')+'.pkl')
        self.time_resolution = time_resolution
        return self.meta


    def create(self, num, model, value):
        next_eid = len(self.entities)
        entities = []

        for i in range(next_eid, next_eid + num):
            eid = f'{self.eid_prefix}{i}'
            self.entities[eid] = value

            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        return time + 1  # Step in smallest time resolution (step size is specified in self.time_resolution)

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            value = self.entities[eid]
            # print(f'get data here: value={value}')
            data[eid] = {}
            for attr in attrs:
                data[eid][attr] = value

        return data


def main():
    pass
    # return mosaik_api_v3.start_simulation(StorageModelSim())


if __name__ == '__main__':
    main()