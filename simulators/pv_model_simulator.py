# simulator_mosaik.py
"""
Mosaik interface for the PV model.

"""
import mosaik_api_v3
from models.PV_model import PV, PVDEFAULTDATANAME
import pandas as pd
import pathlib

META = {
    'type': 'time-based',
    'models': {
        'PVModel': {
            'public': True,
            'params': ['clockfun', 'params_list', 'data_name'],
            'attrs': ['pv_power', 'temp_air'],
        },
    },
}


class PVModelSim(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'PVModel_'
        self.eids = []
        self.pv_time_resolution = 3600 # s

        self.output = pd.DataFrame(columns=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['model', 'attribute']))

        self.data = {}

    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        self.output_path = output_dir.joinpath(self.eid_prefix)
        self.time_resolution = time_resolution
        assert self.pv_time_resolution % time_resolution == 0, 'PV Model expects time resosultion to be an integer fraction of its own time resolution'
        self.int_steps_size = int(self.pv_time_resolution // time_resolution)
        return self.meta

    def create(self, num, model, clockfun, params_list, data_name=PVDEFAULTDATANAME):
        '''
        !!only call once!!!
        create a number of pv models based on PVGIS data
        num: number of models to create
        clockfun: clock function mapping int time steps to datetime
        params_list: list of dicts with parameters for the pv modles to create
        data_name: optional, give the pv data a name to save them as a seperate dataset (for faster access without download)
        '''
        assert len(params_list) == num, 'Number of provided params for PV models does not match num'

        next_eid = len(self.eids)
        entities = []
        
        for i in range(next_eid, next_eid + num):            
            eid = f'{self.eid_prefix}{i}'
            self.eids.append(eid)
            entities.append({'eid': eid, 'type': model})

        self.model = PV(clockfun=clockfun, eids=self.eids, params_list=params_list, data_name=data_name)
        
        return entities

    def step(self, time, inputs, max_advance):
        self.time = time
        assert len(inputs) == 0, 'PV Model expects no inputs'

        datetime, self.data = self.model.step(time)

        # save the outputs
        reshaped_output_dict = {(outerKey, innerKey): values for outerKey, innerDict in self.data.items() for innerKey, values in innerDict.items()}
        for key, values in reshaped_output_dict.items():
            self.output.loc[datetime, key] = values

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


def main():
    return mosaik_api_v3.start_simulation(PVModelSim())

if __name__ == '__main__':
    main()