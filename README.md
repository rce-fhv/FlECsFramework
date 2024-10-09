# Overview
This Framework showcases a method of how to use the MOSAIK framework to model energy communities and evaluate the usefulness of control mechanisms of flexibilities with a specific focus on applicability in practice.
This documentation is not complete and can only be used as a first guide to the framework. It is strongly recommended, to familiarize with the MOSAIK framework first before starting with this project.  
The model and first results obtained with the simulation framework are presented in the following article: https://opus.fhv.at/frontdoor/deliver/index/docId/5442/file/Seiler_Assessing_MPC_for_EC_Flex.pdf . It is recommended, to familiarize with the paper first, to better understand the concepts used. 

Core concept, utilized here is, that Mosaik is used to model dataflows and physical connections as realistically as possible (MOSAIK is not just used as an interface between simulators). Therefore, the scenario can be understood as a model definition and is not unnecessarily cluttered with simulation details, keeping the model and the simulation separated as far as possible. In turn, this puts more responsibilities to the simulators but keeps the scenario cleaner. Therefore, simulation based data storage (for analysis purposes) is implemented at the simulator level.

# Installation
- Clone the git repository into your working directory.
- Create Conda environment
	- Instructions on how to install and create Conda environments can be found on the Conda website.
	- The required packages for the environment "flecsframework_0_1" are contained in the "environment.yml" file, located in the root directory (for windows only).
- Download the required data
	- The required load data for the implemented simulation scenarios is not included in the repository.
	- Users need to download the household load data from the sources under the following link:  https://solar.htw-berlin.de/elektrische-lastprofile-fuer-wohngebaeude/
	- The following dataset is needed: "CSV_74_Loadprofiles_1min_W_var" it needs to be downloaded and the folder needs to be unpacked and placed into the directory "data/raw" in the working directory.
	- The date could also be replaced by custom data, however it is recommended to run the implemented scenario first and start changing from there. 

# Code structure
- main.py
	- Running this file, starts the simulation of all scenarios.
	- Here, the parametric scenario file is called with arguments, to specify the required scenario.
	- The currently implemented scenario allows for the different prediction types and storage model types to be selected.
- scenario
	- This file is called from the main file with the beforementioned arguments.
	- Here, the model energy community is defined based on the input from the main file.
	- Model instances are created and the connections are specified here.
- simulators
	- Simulators are the interface between Mosaik and the models (currently all written in python).
	- They are meant to not handle any model logic but only to provide the interface. 
	- Additionally, the data storage (for the simulation/analysis) is implemented here, to keep the scenario file free from simulation specifics (the output is stored as pickle files).
- Models
	- The models are physical time discrete forward models.
	- They at least implement an init function, where the initial states of the model are set and a step function, which evolves the model through time.
- Postprocessing and Analysis
	- The directory post prostprocessing_analysis provides two Jupyter notebooks for the analysis of the results.
	- Single scenarios can be analyzed and a comparison of different scenarios is possible. 

# Running the first example
- To run the simulation, simply run main.py after the installation procedure.
- Running all scenario takes a signifikant amount of time (several hours) and ram space, therefore one might want to exclude some of the scenarios by removing them from the prediction types and storage_model_types lists in the main.py file. 
- Evaluating the results:
	- The results are stored as picke files in the data/output directory. 
	- Jupyter notebook are used for the evaluation of the results.
	- A single scenario can be analyzed by running the following notebook and observing the output and figures: "\postprocessing_analysis\analysis_single_scenario.ipynb".

# Citing
If you use this framework in the academic context, please consider citing the following article: 
https://opus.fhv.at/frontdoor/deliver/index/docId/5442/file/Seiler_Assessing_MPC_for_EC_Flex.pdf published at the e-nova conference 2024 (https://doi.org/10.57739/978-3-903207-89-9)
