# simulator_mosaik.py
"""
Mosaik interface for the MPC battery storage controller.

"""
import pandas as pd
import mosaik_api_v3
from models.controller.MPC_battery_storage_controller import MPCBatteryStorageController
import pathlib


META = {
    'type': 'time-based',
    'models': {
        'MPController': {
            'public': True,
            'params': ['initial_controll_power', 'clockfun', 'milp_battery_model', 'predictor', 'warmup_time'],
            'attrs': ['storage_actual_capacity', 'storage_actual_power', 'controll_power', 'historic_load_data', 'pv_power', 'temp_air'],
        },
    },
}

class MPControllerSim(mosaik_api_v3.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = 'MPController_'
        self.entities = {}  # Maps EIDs to model instances/entities
        self.time = -1

        self.output = pd.DataFrame(index=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['time', 'prediction_time']), columns=pd.MultiIndex(levels=[[], []], codes=[[], []], name=['model', 'attribute']))

    def init(self, sid, time_resolution, eid_prefix=None, output_dir='output/default'):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True) # create output directory if it does not exist
        self.output_path = output_dir.joinpath(self.eid_prefix)
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, clockfun, initial_controll_power, milp_battery_model, predictor, warmup_time):
        next_eid = len(self.entities)
        entities = []

        for i in range(next_eid, next_eid + num):
            model_instance = MPCBatteryStorageController(clockfun=clockfun, initial_controll_power=initial_controll_power, delta_t=self.time_resolution,  milp_battery_model=milp_battery_model, predictor=predictor, warmup_time=warmup_time)
            eid = f'{self.eid_prefix}{i}'
            self.entities[eid] = model_instance
            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        assert time - self.time == 1, 'Something is off with the timing in the MPController'
        self.time = time

        for eid, model_instance in self.entities.items():
            attrs = inputs[eid]
            assert len(attrs['storage_actual_capacity']) == 1, 'MPCController expects only one input for storage_capacity'
            [_, storage_capacity], = attrs.pop('storage_actual_capacity').items() # unpack mosaik dict structure
            
            assert len(attrs['storage_actual_power']) == 1, 'MPCController expects only one input for storage_power'
            [_, storage_power],    = attrs.pop('storage_actual_power').items()  # unpack mosaik dict structure
            
            [_, historic_load_data], = attrs.pop('historic_load_data', {'NoDSO': None}).items()  # get historic load data from input, if not available, return None, then unpack mosaik dict structure

            pv_power = sum(attrs.pop('pv_power', 0).values())  # get pv data from input and sum up the values

            [_, temp_air], = attrs.pop('temp_air', {'NoPV': None}).items()  # get temp_air from input, if not available, return None, then unpack mosaik dict structure        

            datetime, _ = model_instance.step(time, capacity_sorage_tminus1=storage_capacity, power_storage_tminus1=storage_power, historic_load_data=historic_load_data, pv_power=pv_power, temp_air=temp_air)

            # TODO: Implement saving of prediction values
            # df_steps = model_instance.milp_battery_model.output_df_steps
            # self.milp_battery_model.output_df_times
            # self.output.loc[datetime, (eid, list(df_steps.columns), list(df_steps.index))] = df_steps.values.T

            # save the outputs
            
            for k, v in model_instance.current_prediction.items(): # Maybe improve this for speed! Multiindexing in this way is difficult and might not be possible at all!
                self.output.loc[(datetime, k), (eid, 'Residual Load Prediction')] = v

            for k, v in model_instance.storage_controll_power_horizon.items(): # Maybe improve this for speed! Multiindexing in this way is difficult and might not be possible at all!
                self.output.loc[(datetime, k), (eid, 'Control Power')] = v    

            for k, v in model_instance.storage_controll_capacity_horizon.items(): # Maybe improve this for speed! Multiindexing in this way is difficult and might not be possible at all!
                self.output.loc[(datetime, k), (eid, 'Control Capacity')] = v

            for k, v in model_instance.storage_controll_capacity1_horizon.items(): # Maybe improve this for speed! Multiindexing in this way is difficult and might not be possible at all!
                self.output.loc[(datetime, k), (eid, 'Control Capacity 1')] = v

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
    return mosaik_api_v3.start_simulation(MPControllerSim())


if __name__ == '__main__':
    main()