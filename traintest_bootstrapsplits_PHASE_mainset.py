from SocialGNN_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline
import os, sys
import pickle
import numpy as np
from collections import namedtuple

### LOAD DATA
model_to_train = sys.argv[1]
bootstrap_splits = 10
string = './TrainedModels/PHASE_mysplit_humanratings_Jul23_'

## Get videos data (position, vel, landmark info etc and associated human rating labels)
with open('./PHASE/Videos_humanratings', "rb") as f:
  Videos = pickle.load(f)

mapping = {'friendly':(1,0,0), 'neutral': (0,1,0), 'adversarial': (0,0,1)}


# Start Training

if model_to_train == "SocialGNN_V" or model_to_train == "SocialGNN_V_onlyagents":
  print("\n.............BOOTSTRAPPING TRAIN/TEST SPLIT..............")

  model_config = namedtuple('model_config', 'NUM_NODES NUM_AGENTS V_SPATIAL_SIZE E_SPATIAL_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

  for split in range(bootstrap_splits):
    # Load data for current bootstrap (X_train, X_test contain indices of Videos, and y_train, y_test contain human rating labels)
    with open('./PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_'+str(split), "rb") as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
    print("\n\n Bootstrap No.:",split)

    # change parameters for only agent edges
    C = model_config(NUM_NODES = 4, NUM_AGENTS = 2, V_SPATIAL_SIZE = 12, E_SPATIAL_SIZE = 12, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 150
    model = SocialGNN(Videos, C, sample_graph_dicts_list)
    model._initialize_session()
    scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
    scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
    scores['testset'].append(model.test(test_data_idx=X_test, mapping=mapping))
    print(scores)

    model.save_model(string  + str(split) + '_' + model_to_train)
    

elif model_to_train == "SocialGNN_E" or model_to_train == "SocialGNN_E_onlyagentedges":
  print("\n.............BOOTSTRAPPING TRAIN/TEST SPLIT..............")

  model_config = namedtuple('model_config', 'NUM_NODES MAX_EDGES E_SPATIAL_SIZE E_TEMPORAL_SIZE E_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

  for split in range(bootstrap_splits):
    # Load data for current bootstrap (X_train, X_test contain indices of Videos, and y_train, y_test contain human rating labels)
    with open('./PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_'+str(split), "rb") as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
    print("\n\n Bootstrap No.:",split)

    # change parameters for only agent edges
    C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 12, E_TEMPORAL_SIZE = 6, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 150
    model = SocialGNN_E(Videos, C, sample_graph_dicts_list)
    model._initialize_session()
    scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
    scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
    scores['testset'].append(model.test(test_data_idx=X_test, mapping=mapping))
    print(scores)

    model.save_model(string + str(split) + '_' + model_to_train)

elif model_to_train == "CueBasedLSTM":
  #CueBasedLSTM-Relation: set 28-->40 explicit_edges=True
  print("\n.............BOOTSTRAPPING TRAIN/TEST SPLIT..............")

  model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

  scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

  for split in range(bootstrap_splits):
    # Load data for current bootstrap (X_train, X_test contain indices of Videos, and y_train, y_test contain human rating labels)
    with open('./PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_'+str(split), "rb") as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
    print("\n\n Bootstrap No.:",split)

    C = model_config(FEATURE_SIZE = 28, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 150
    model = CueBasedLSTM(Videos, C, explicit_edges = False)
    model._initialize_session()
    scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
    scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
    scores['testset'].append(model.test(test_data_idx=X_test, mapping=mapping))
    print(scores)

    model.save_model(string + str(split) + '_' + model_to_train)
    
