from SocialGNN_withU_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline
import os, sys
import pickle
import numpy as np
from collections import namedtuple
import time
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help="SocialGNN_V/SocialGNN_E/CueBasedLSTM/CueBasedLSTM-Relation", type= str)
parser.add_argument('--context_info', help="True/False", type=lambda x: bool(strtobool(x)), default=False)

args = parser.parse_args()
timestr = time.strftime("%Y%m%d")

### LOAD DATA
bootstrap_splits = 10
#string = './TrainedModels/PHASE_mysplit_humanratings_Jul23_'
model_string = './TrainedModels/PHASE_mysplit_humanratings_context'+ str(args.context_info) + '_' + timestr + '_'

## Get videos data (position, vel, landmark info etc and associated human rating labels)
with open('./PHASE/Videos_humanratings', "rb") as f:
  Videos = pickle.load(f)

mapping = {'friendly':(1,0,0), 'neutral': (0,1,0), 'adversarial': (0,0,1)}


# Start Training

if args.model_name == "SocialGNN_V" or args.model_name == "SocialGNN_V_onlyagents":
  print("\n.............BOOTSTRAPPING TRAIN/TEST SPLIT..............")

  model_config = namedtuple('model_config', 'NUM_NODES NUM_AGENTS V_SPATIAL_SIZE E_SPATIAL_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

  a = 2 if args.model_name == "SocialGNN_V_onlyagents" else 4

  for split in range(bootstrap_splits):
    # Load data for current bootstrap (X_train, X_test contain indices of Videos, and y_train, y_test contain human rating labels)
    with open('./PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_'+str(split), "rb") as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
    print("\n\n Bootstrap No.:",split)

    # change parameters for only agent edges
    C = model_config(NUM_NODES = 4, NUM_AGENTS = a, V_SPATIAL_SIZE = 64, E_SPATIAL_SIZE = 64, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 3
    model = SocialGNN(Videos, C, args.context_info, sample_graph_dicts_list)
    model._initialize_session()
    scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
    scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
    scores['testset'].append(model.test(test_data_idx=X_test, mapping=mapping))
    print(scores)

    model.save_model(model_string  + str(split) + '_' + args.model_name)
    

elif args.model_name == "SocialGNN_E" or args.model_name == "SocialGNN_E_onlyagentedges":
  print("\n.............BOOTSTRAPPING TRAIN/TEST SPLIT..............")

  model_config = namedtuple('model_config', 'NUM_NODES MAX_EDGES E_SPATIAL_SIZE E_TEMPORAL_SIZE E_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

  a = True if args.model_name == "SocialGNN_E_onlyagentedges" else False

  for split in range(bootstrap_splits):
    # Load data for current bootstrap (X_train, X_test contain indices of Videos, and y_train, y_test contain human rating labels)
    with open('./PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_'+str(split), "rb") as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
    print("\n\n Bootstrap No.:",split)

    # change parameters for only agent edges
    C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 64, E_TEMPORAL_SIZE = 16, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 3
    model = SocialGNN_E(Videos, C, args.context_info, sample_graph_dicts_list, ablate = a)
    model._initialize_session()
    scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
    scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
    scores['testset'].append(model.test(test_data_idx=X_test, mapping=mapping))
    print(scores)

    model.save_model(model_string + str(split) + '_' + args.model_name)

elif args.model_name == "CueBasedLSTM" or args.model_name == "CueBasedLSTM-Relation":
  print("\n.............BOOTSTRAPPING TRAIN/TEST SPLIT..............")

  model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

  scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

  if args.model_name == "CueBasedLSTM":
    f_size = 52 if args.context_info==True else 28
    e = False
  else:
    f_size = 64 if args.context_info==True else 40
    e = True

  for split in range(bootstrap_splits):
    # Load data for current bootstrap (X_train, X_test contain indices of Videos, and y_train, y_test contain human rating labels)
    with open('./PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_'+str(split), "rb") as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
    print("\n\n Bootstrap No.:",split)

    C = model_config(FEATURE_SIZE = f_size, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 3
    model = CueBasedLSTM(Videos, C, args.context_info, explicit_edges = e)
    model._initialize_session()
    scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
    scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
    scores['testset'].append(model.test(test_data_idx=X_test, mapping=mapping))
    print(scores)

    model.save_model(model_string + str(split) + '_' + args.model_name)
    
