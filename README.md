# GANUrbanMorph

### ReadMe for Python and R Code used in the GAN Evaluation Study
### Author: Bhartendu Pandey
### Directory Structure:
```
Root
├── 1_TrainingandTestingDataCollection
│   ├── 1_CreateTesting_seed_number.py
│   └── 1_CreateTraining.py
├── 2_ModelTraining
│   ├── TrainBFBHModel_100lambda_Multiple_v1.py
│   └── TrainLULCBFModel_100lambda_Multiple_v1.py
├── 3_GANValidation
│   ├── GeneratorDiscrimatorLoss.py
│   ├── Validation_BF.py
│   └── Validation_BH.py
├── 4_Inference
│   ├── Generator.py
│   └── InferenceSingleLULC.py
├── 5_UrbanScaling
│   ├── BuildingVolume_Heterogeneity_LAPlot.py
│   ├── BuildingVolume_Population.R
│   └── exportmsalevelbuildingsdata.py
├── BPspatlibv0.py
├── BPspatlibv1.py
└── buildingspoly_CApy.py
```
### Description:

1_TrainingandTestingDataCollection: This folder contains the python scripts, i.e., 1_CreateTraining.py and 1_CreateTesting_seed_number.py, to generate training data (8,000 tiles) and testing data (2,000 tiles), respectively.
2_ModelTraining: This folder contains two python scripts to train generative adversarial networks for building footprints generation (TrainLULCBFModel_100lambda_Multiple_v1.py) and building heights generation (TrainBFBHModel_100lambda_Multiple_v1.py). Note that these script uses the outputs from scripts under 1_TrainingandTestingDataCollection.

3_GANValidation: This folder contains three python scripts.
•	GeneratorDiscrimatorLoss.py uses csv outputs, from TrainLULCBFModel_100lambda_Multiple_v1.py and TrainBFBHModel_100lambda_Multiple_v1.py, to plot generator loss.
•	Validation_BF.py script uses trained model (GAN) from TrainLULCBFModel_100lambda_Multiple_v1.py, run inference on the test data generated from 1_CreateTesting_seed_number.py, and plots/prints the outputs including error and bias metrics.
•	Similarly. Validation_BH.py scripts uses trained model (GAN)from TrainBFBHModel_100lambda_Multiple_v1.py, run inference on the test data generated from 1_CreateTesting_seed_number.py, and plots/prints the outputs including error and bias metrics.

4_Inference: This folder contains two python scripts: Generator.py and InferenceSingleLULC.py. The former is a script containing helper function to run inference and the latter runs inferences and analyzes the GAN outputs for pixels that became developed between 2010 and 2020.

5_UrbanScaling: This folder contains three python script and two scripts. 
•	exportmsalevelbuildingsdata.py script generates gpkg files for the Model America dataset (originally in csv format for individual states and converted to GPKG using CSVProcessing.R) at the metropolitan statistical area (MSA) scale for all MSAs in the US (exluding Puerto Rico).
•	 Next, BuildingVolume_Population.R script runs building volume and population scaling analysis across MSAs and for Los Angeles urban area using Model America Data and the GAN output obtained from InferenceSingleLULC.py [land change mask data and building volume data].
•	buildingspoly_CApy.py converts Model America dataset for the state of California (originally in csv format) to GPKG format.
•	BuildingVolume_Heterogeneity_LAPlot.py: This script runs Taylor’s power law scaling analysis for the Los Angeles urban area.

BPspatlibv0.py and BPspatlibv1.py contains helper functions referenced in some of the scripts above.
