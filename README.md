# SocialGNN: A Graph Neural Network model for Social Interaction Recognition 
[![DOI](https://zenodo.org/badge/561537272.svg)](https://zenodo.org/badge/latestdoi/561537272)
<!--
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip) ![Progress](https://progress-bar.dev/85/?title=completed) -->

This project is described in our paper titled "Relational visual representations underlie human social interaction recognition". [Link to Paper](https://www.nature.com/articles/s41467-023-43156-8)

This repository contains code to train and test our SocialGNN models (as well as baseline VisualRNN models) on animated and natural stimuli.

P.S.: VisualRNN = CueBasedLSTM

## Conda Environment / Prerequisites
- For macOS: 
```Conda Environments/condaenv_macbook_gnnEnv_Oct17_2022.yml ```
- For Linux: ```Conda Environments/condaenv_rockfish_gnnEnv_Jul1423.yml ```

## How to Run the Code?
Please refer to our paper for the terminology used here and for changes in parameter settings.

#### Running SocialGNN on the PHASE standard set (400 videos)

###### Getting Accuracy and Predicted Labels using Trained Models
  ```
  python get_accuracy_predictions_PHASE_mainset.py --model_name=SocialGNN_E --train_datetime=20230503 --context_info=True --bootstrap_no=0 --save_predictions=False
  ```
<sup>--model_name= SocialGNN_V/ SocialGNN_E/ CueBasedLSTM/ CueBasedLSTM-Relation/ SocialGNN_V_onlyagents/ SocialGNN_E_onlyagentedges </sup> \
<sup> --train_datetime= 20230503 / 20230617 (for SocialGNN_V_onlyagents/SocialGNN_E_onlyagentedges) </sup> \
<sup> --bootstrap_no= 0-9 </sup>
  
###### Training (with bootstrapped train-test splits) SocialGNN or VisualRNN/CueBasedLSTM models
  ```
  python traintest_bootstrapsplits_PHASE_mainset.py --model_name="SocialGNN_V" --context_info=True
  ```
#### Running SocialGNN on the PHASE generalization set (100 videos)
###### Getting Accuracy using Trained Models
  ```
  python traintest_PHASE_genset.py --mode=test --model_name=SocialGNN_E --context_info=True
  ```
###### Training SocialGNN or VisualRNN/CueBasedLSTM models
```
python traintest_PHASE_genset.py --mode=train --model_name=SocialGNN_E --context_info=True
```

#### RSA on the PHASE datasets
###### Getting Model Representations using Trained Models
```
python SocialGNN_get_activations.py --model_name=SocialGNN_E --context_info=True --bootstrap_no=0 --dataset=main_set --train_datetime=20230503 --activation_type=RNN
```
```
python SocialGNN_get_activations.py --model_name=SocialGNN_E --context_info=True --dataset=generalization_set --train_datetime=20230515 --activation_type=RNN
```
###### Creating RDMs and RSA
Note: motion energy files need to be downloaded into 'Activations' from OSF folder
```
RSA_Github.ipynb
```

#### Running SocialGNN on the Gaze dataset
Set <prediction_type> to 2 for social v/s non-social classification; set to 5 for classifying into the 5 gaze labels
The first 10 bootstrpas correspond to "dataset=5Jun23", the next 10 to "dataset=14Jun23"

###### Getting Accuracy (on all bootstrapped train-test splits) using Trained Models
  ```
  python traintest_bootstrapsplits_Gaze.py test <model_to_test> <prediction_type> <dataset>
  
  Example:
  python traintest_bootstrapsplits_Gaze.py test CueBasedLSTM-Relation 5 5Jun23
  ```
###### Training SocialGNN or VisualRNN/CueBasedLSTM models
```
python traintest_bootstrapsplits_Gaze.py train <model_to_train> <prediction_type> <dataset>
```
###### Training/Testing VGG19 model
```
python VGG19full_traintest_gaze.py --mode=test --dataset=5Jun23 --output_type=2
```

#### Colab Notebook for Plots
Note 1: may need to run this outside gnnEnv conda environment
Note 2: need to download original .pik files from the PHASE dataset to rerun all
```
SocialGNN_Generating_Plots_Github.ipynb
```
<!--- ## Repository Components --->


## Citation
```
Malik, M., Isik, L. Relational visual representations underlie human social interaction recognition. Nat Commun 14, 7317 (2023)
```

### For questions/issues: 👩‍💻 mmalik16@jhu.edu
