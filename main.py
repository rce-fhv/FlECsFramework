# from framework.mosaik.scenario_first_test import run_scenario_first_testconda i
from scenarios.scenario_variable_storage_and_prediction import run_scenario_variable_storage_and_prediction
from datetime import datetime

# run_scenario_first_test()

prediction_types = ['PerfectPrediction']  # ,'PerfectPrediction', 'PersistencePrediction',  'PersistencePrediction1', LSTMLoadPerfectPvPrediction]
storage_model_types = ['StorageModelPerfect']  # , 'StorageModelReducedMaxCap', 'StorageModelSelfdischargeReducedMaxCap']  # ['StorageModelPerfect', 'StorageModelReducedMaxCap', StorageModelSelfdischarge, StorageModelSelfdischargeReducedMaxCap

starttime = datetime.now().strftime("%Y%m%d%H%M")

i=0
for predictor in prediction_types:
    for storage_model in storage_model_types:
        print(i)
        print(predictor, storage_model)
        i+=1
        # try:
        output_dir = f'data/output/output_{starttime}/{predictor}_and_{storage_model}'
        run_scenario_variable_storage_and_prediction(predictor_type=predictor, storage_model_type=storage_model, output_dir=output_dir)
        # except Exception as e: 
        #     print(f'The following error has occured at combination {predictor}, {storage_model}, {e}')