<img src="icon.png" align="right" />

# SocialGNN: A Graph Neural Network model for Social Interaction Recognition 

The project is described in our paper titled "Relational Visual Information explains Human Social Inference: A Graph Neural Network model for Social Interaction Recognition". [Link to Preprint](https://psyarxiv.com/5cuyr)

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


## How to run this code?

#### Conda Environment / Prerequisites

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

## Repository Components

## Citation
