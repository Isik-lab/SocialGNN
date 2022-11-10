<img src="icon.png" align="right" />

# SocialGNN: A Graph Neural Network model for Social Interaction Recognition 

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip) ![Progress](https://progress-bar.dev/75/?title=completed)

This project is described in our paper titled "Relational Visual Information explains Human Social Inference: A Graph Neural Network model for Social Interaction Recognition". [Link to Preprint](https://psyarxiv.com/5cuyr)

This repository contains code to train and test our SocialGNN models (as well as baseline VisualRNN models) on animated and natural stimuli.

P.S.: VisualRNN = CueBasedLSTM

## Conda Environment / Prerequisites
- For macOS: 
```Conda Environment/condaenv_macbook_gnnEnv_Oct17_2022.yml ```
- For Linux: To Be Added

## How to Run the Code?
Please refer to our paper for the terminology used here and for changes in parameter settings.

#### Running SocialGNN on the PHASE standard set (400 videos)

###### Getting Accuracy and Predicted Labels using Trained Models
  ```
  python get_accuracy_predictions_PHASE_mainset.py <bootstrap no> <model_to_test>
  
  Example:
  python get_accuracy_predictions_PHASE_mainset.py 3 SocialGNN_E
  ```
###### Training (with bootstrapped train-test splits) SocialGNN or VisualRNN/CueBasedLSTM models
  ```
  python traintest_bootstrapsplits_PHASE_mainset.py <model_to_train>
  ```
#### Running SocialGNN on the PHASE generalization set (100 videos)
###### Getting Accuracy using Trained Models
  ```
  python traintest_PHASE_genset.py test <model_to_test>
  
  Example:
  python traintest_PHASE_genset.py test SocialGNN_E
  ```
###### Training SocialGNN or VisualRNN/CueBasedLSTM models
```
  python traintest_PHASE_genset.py train <model_to_train>
```


#### Running SocialGNN on the Gaze dataset
Set <prediction_type> to 2 for social v/s non-social classification; set to 5 for classifying into the 5 gaze labels

###### Getting Accuracy (on all bootstrapped train-test splits) using Trained Models
  ```
  python traintest_bootstrapsplits_Gaze.py test <model_to_test> <prediction_type>
  
  Example:
  python traintest_bootstrapsplits_Gaze.py test CueBasedLSTM-Relation 5
  ```
###### Training SocialGNN or VisualRNN/CueBasedLSTM models
```
  python traintest_bootstrapsplits_Gaze.py train <model_to_train> <prediction_type>
```

<!--- ## Repository Components --->

## To Be Added Soon
- Data Preprocessing files
- Gaze dataset processed bootstrap split files (too big for github)

## Citation
```
```

### For Issues: 👩‍💻 mmalik16@jhu.edu
