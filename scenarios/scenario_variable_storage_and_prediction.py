import mosaik
import mosaik.util

import sys
import numpy as np
import pandas as pd

# sys.path.append('../../')
from models.clock import clock
from models.controller.forecast.predictors import PerfectResidualPowerPrediction, PersistenceResidualPowerPrediction
from models.controller.MPC_battery_storage_controller import MILPBatteryModel
from models.controller.forecast.predictors import PerfectResidualPowerPrediction, PersistenceResidualPowerPrediction, PersistenceResidualPowerPrediction1, LoadForecastWithPerfectPvPrediction
from models.clock import clock


def run_scenario_variable_storage_and_prediction(predictor_type, storage_model_type, output_dir):  # storage_model, predictor

    # Sim config. and other parameters
    SIM_CONFIG = {
        'LoadModelSim': {
            'python': 'simulators.load_model_simulator:LoadModelSim',
        },
        'PVModelSim': {
            'python': 'simulators.pv_model_simulator:PVModelSim',
        },
        'GridSim': {
            'python': 'simulators.grid_model_simulator:GridSim',
        },
        'StorageModelSim':{
            'python': 'simulators.battery_storage_model_simulator:StorageModelSim',
        },
        'MPCSim':{
            'python': 'simulators.mpc_storage_controller_simulator:MPControllerSim',
        },
        'DSOSim':{
            'python': 'simulators.dso_model_simulator:DSOModelSim',
        },
        'PODSim':{
            'python': 'simulators.pod_simulator:PODSim',
        }
    }

    TIME_RESOLUTION = 3600 # s/step
    # test setup
    # END = 24*7   # 8760  #  #  #  75 *24  # 8760 + 65*24  # 65 *24 #   #   # 30*24 # 8760 # *3600  # 2 h  48 #
    # WARMUPTIME = pd.Timedelta(days=3)  # 60   3
    # clockfun = clock(TIME_RESOLUTION, origin_timestamp='2010-03-22 00:00:00')
    
    # real setup
    END = 8760 + 65*24  # 24*7  #  #  75 *24  # 8760 + 65*24  # 65 *24 #   #   # 30*24 # 8760 # *3600  # 2 h  48 #
    WARMUPTIME = pd.Timedelta(days=60)  # 60   3
    clockfun = clock(TIME_RESOLUTION, origin_timestamp='2009-11-01 01:00:00') # start at beginning of november for 2 months of pretraining


    # Create World
    world = mosaik.World(SIM_CONFIG, time_resolution=TIME_RESOLUTION, debug=False)  # TIME_RESOLUTION)

    # Start simulators
    loadsim = world.start('LoadModelSim', output_dir=output_dir)
    pvsim   = world.start('PVModelSim', output_dir=output_dir)
    gridsim = world.start('GridSim', output_dir=output_dir)
    bessim  = world.start('StorageModelSim', output_dir=output_dir)
    mpcsim  = world.start("MPCSim", output_dir=output_dir)
    dsosim  = world.start("DSOSim", output_dir=output_dir)
    podsim  = world.start("PODSim", output_dir=output_dir)

    # Instantiate models
    load_models = loadsim.LoadModel.create(74, # 74
                                           clockfun=clockfun,
                                           )
    pv_models   = pvsim.PVModel.create(3, 
                                       clockfun=clockfun, 
                                       params_list=[
                                            dict(
                                                latitude  = 47.40631251493437, # Standort FHV, In decimal degrees, between -90 and 90, north is positive (ISO 19115)
                                                longitude = 9.744656758087116, # Standort FHV, In decimal degrees, between -180 and 180, east is positive (ISO 19115)
                                                start = '2009',                # SEVA: time index has to start one year before the relevant year due to time zone conversion
                                                end   = '2011',                # SEVA: time index goes until '2007-01-01 00:00:00+01:00'
                                                surface_azimuth = 180,         # Orientation (azimuth angle) of the (fixed) plane. Clockwise from north (north=0, east=90, south=180, west=270)
                                                usehorizon = True,             # Include effects of horizon
                                                pvcalculation = True,          # Return estimate of hourly PV production
                                                peakpower = 125,               # Nominal power of PV system in kW
                                                pvtechchoice = 'crystSi',      # ({'crystSi', 'CIS', 'CdTe', 'Unknown'}, default: 'crystSi')
                                                mountingplace = 'building',    # ({'free', 'building'}, default: free) – Type of mounting for PV system. Options of ‘free’ for free-standing and ‘building’ for building-integrated.
                                                loss = 15,                     # (float, default: 0) – Sum of PV system losses in percent. Required if pvcalculation=True
                                                optimal_surface_tilt = True,   # (bool, default: False) – Calculate the optimum tilt angle. Ignored for two-axis tracking
                                                additional_outputs=['temp_air']
                                                ),
                                            dict(
                                                latitude  = 47.40631251493437, # Standort FHV, In decimal degrees, between -90 and 90, north is positive (ISO 19115)
                                                longitude = 9.744656758087116, # Standort FHV, In decimal degrees, between -180 and 180, east is positive (ISO 19115)
                                                start = '2009',                # SEVA: time index has to start one year before the relevant year due to time zone conversion
                                                end   = '2011',                # SEVA: time index goes until '2007-01-01 00:00:00+01:00'
                                                surface_azimuth = 90,          # Orientation (azimuth angle) of the (fixed) plane. Clockwise from north (north=0, east=90, south=180, west=270)
                                                usehorizon = True,             # Include effects of horizon
                                                pvcalculation = True,          # Return estimate of hourly PV production
                                                peakpower = 62.5,              # Nominal power of PV system in kW
                                                pvtechchoice = 'crystSi',      # ({'crystSi', 'CIS', 'CdTe', 'Unknown'}, default: 'crystSi')
                                                mountingplace = 'building',    # ({'free', 'building'}, default: free) – Type of mounting for PV system. Options of ‘free’ for free-standing and ‘building’ for building-integrated.
                                                loss = 15,                     # (float, default: 0) – Sum of PV system losses in percent. Required if pvcalculation=True
                                                optimal_surface_tilt = True,   # (bool, default: False) – Calculate the optimum tilt angle. Ignored for two-axis tracking
                                                ),
                                            dict(
                                                latitude  = 47.40631251493437, # Standort FHV, In decimal degrees, between -90 and 90, north is positive (ISO 19115)
                                                longitude = 9.744656758087116, # Standort FHV, In decimal degrees, between -180 and 180, east is positive (ISO 19115)
                                                start = '2009',                # SEVA: time index has to start one year before the relevant year due to time zone conversion
                                                end   = '2011',                # SEVA: time index goes until '2007-01-01 00:00:00+01:00'
                                                surface_azimuth = 270,         # Orientation (azimuth angle) of the (fixed) plane. Clockwise from north (north=0, east=90, south=180, west=270)
                                                usehorizon = True,             # Include effects of horizon
                                                pvcalculation = True,          # Return estimate of hourly PV production
                                                peakpower = 62.5,              # Nominal power of PV system in kW
                                                pvtechchoice = 'crystSi',      # ({'crystSi', 'CIS', 'CdTe', 'Unknown'}, default: 'crystSi')
                                                mountingplace = 'building',    # ({'free', 'building'}, default: free) – Type of mounting for PV system. Options of ‘free’ for free-standing and ‘building’ for building-integrated.
                                                loss = 15,                     # (float, default: 0) – Sum of PV system losses in percent. Required if pvcalculation=True
                                                optimal_surface_tilt = True,   # (bool, default: False) – Calculate the optimum tilt angle. Ignored for two-axis tracking
                                                )
                                            ]
                                        )
    match storage_model_type:
        case 'StorageModelPerfect': 
            bes_model = bessim.StorageModel(capacity_0=50_000*3600, 
                clockfun=clockfun,
                battery_params={
                    'max_capacity':250_000*3600,
                    'max_power_discharge': 250_000,
                    'max_power_charge': 250_000
                    })
        case 'StorageModelReducedMaxCap': # dont run this for enova paper
            bes_model = bessim.StorageModel(capacity_0=50_000*3600, 
                clockfun=clockfun,
                battery_params={
                    'max_capacity':250_000*3600*0.8, # max capacity reduced to 80%
                    'max_power_discharge': 250_000,
                    'max_power_charge': 250_000
                    })
        case 'StorageModelSelfdischarge':
            bes_model = bessim.StorageModelWithSelfdischarge(
                capacity_0=50_000*3600, 
                clockfun=clockfun,
                battery_params={
                    'max_capacity':250_000*3600,
                    'max_power_discharge': 250_000,
                    'max_power_charge': 250_000
                    })
        case 'StorageModelSelfdischargeReducedMaxCap':
            bes_model = bessim.StorageModelWithSelfdischarge(
                capacity_0=50_000*3600, 
                clockfun=clockfun,
                battery_params={
                    'max_capacity':250_000*3600*0.8, # max capacity reduced to 80%
                    'max_power_discharge': 250_000,
                    'max_power_charge': 250_000
                    })
        case _:
            raise KeyError(f'Non existant option {storage_model_type}')

    milp_model = MILPBatteryModel(objective_key='selfconsumption', # min_max_power # selfconsumption # price_driven selfsufficiency
                                  max_capacity=250_000*3600,
                                  max_power_discharge=250_000,
                                  max_power_charge=250_000
                                  )
    
    match predictor_type:
        case 'PerfectPrediction':
            predictor = PerfectResidualPowerPrediction(clockfun, horizon_intsteps=24)
        case 'PersistencePrediction':  
            predictor = PersistenceResidualPowerPrediction(clockfun, horizon_intsteps=24)
        case 'PersistencePrediction1':  
            predictor = PersistenceResidualPowerPrediction1(clockfun, horizon_intsteps=24)
        case 'LSTMLoadPerfectPvPrediction':  
            predictor = LoadForecastWithPerfectPvPrediction(clockfun, horizon_intsteps=24, limit_history=pd.Timedelta(days=60))
        case _:
            raise KeyError(f'Non existant option {storage_model_type}')

    mpc_contr   = mpcsim.MPController(clockfun=clockfun, 
                                      initial_controll_power=0,
                                      milp_battery_model=milp_model, 
                                      predictor=predictor,
                                      warmup_time=WARMUPTIME
                                      )
    grid_model  = gridsim.GridModel(clockfun=clockfun)
    dso_model   = dsosim.DSOModel(clockfun=clockfun)
    pod_models  = podsim.PODModel.create(76, clockfun=clockfun)


    # Connect entities
    # Connect all consumers, producers and flexibilities to a POD

    # In front of the meter ("Volleinspeiser")

    # connect the first PV model (south-facing) to a randomly chosen POD
    pod_pv = np.random.choice(pod_models, 1, replace=False)[0]
    # pod_pv = pod_models[0]
    world.connect(pv_models[0], pod_pv, ('pv_power', 'btm_power'))

    # connect the first PV model (south-facing) to the controller to send the weather data

    
    # connect the storage to a random POD
    pod_bes = np.random.choice([m for m in pod_models if m not in [pod_pv]], 1, replace=False)[0]
    # pod_bes = pod_models[1]
    world.connect(bes_model, pod_bes, ('actual_power', 'btm_power'))
    
    # connect the loadas to all remaining POD's
    pod_load = [m for m in pod_models if m not in [pod_pv]+[pod_bes]]
    for load_m, pod_m in zip(load_models, pod_load):
        world.connect(load_m, pod_m, ('load', 'btm_power'))
        
    # connect the other 2 PV systems to the selected POD's ("Überschusseinspeiser")
    pod_pv_load = np.random.choice(pod_load, 2, replace=False)
    # pod_pv_load = pod_load[2:4]
    for pv_m, pod_m in zip(pv_models[1:], pod_pv_load): 
        world.connect(pv_m, pod_m, ('pv_power', 'btm_power'))

    # connect pv systems to the controller
    # connect pv power to the controller
    mosaik.util.connect_many_to_one(world, pv_models, mpc_contr, ('pv_power', 'pv_power'))
    # get the weather data from the pv to the controller
    world.connect(pv_models[0], mpc_contr, ('temp_air', 'temp_air'))
    
    # connect all POD's to the grid and the DSO
    mosaik.util.connect_many_to_one(world, pod_models, grid_model, ('grid_power', 'node_power'))
    mosaik.util.connect_many_to_one(world, pod_models, dso_model, ('grid_power', 'node_power'))

    # Connect Storage and controller (both ways)
    world.connect(mpc_contr, bes_model, ('controll_power', 'power'), time_shifted=True, initial_data={'controll_power':0})
    world.connect(bes_model, mpc_contr, ('capacity', 'storage_actual_capacity'), ('actual_power', 'storage_actual_power'), time_shifted=True, initial_data={'capacity':0, 'actual_power':0})

    # connect DSO and EC controller to send the historic data
    world.connect(dso_model, mpc_contr, ('node_data', 'historic_load_data'))


    # Run simulation
    world.run(until=END, print_progress=True)

    world.shutdown()
 
    # mosaik.util.plot_execution_graph(world, folder='util_figures')
    # mosaik.util.plot_dataflow_graph(world, folder='')