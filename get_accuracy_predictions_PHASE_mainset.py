from SocialGNN_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline

import os, sys
import pickle
import numpy as np
from collections import namedtuple

### LOAD DATA
bootstrap_no = sys.argv[1]
model_to_test = sys.argv[2]
string = './TrainedModels/PHASE_mysplit_humanratings_May1_' + bootstrap_no + '_'
bootstrap_file = './PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_' + bootstrap_no


## Get videos data (position, vel, landmark info etc and associated human rating labels)
with open('./PHASE/Videos_humanratings', "rb") as f:
  Videos = pickle.load(f)

mapping = {'friendly':(1,0,0), 'neutral': (0,1,0), 'adversarial': (0,0,1)}

# Load data for current bootstrap (X_train, X_test contain indices of Videos, and y_train, y_test contain human rating labels)
with open(bootstrap_file, "rb") as f:
  X_train = pickle.load(f)
  y_train = pickle.load(f)
  X_test = pickle.load(f)
  y_test = pickle.load(f) 

# Start Training and Testing

if model_to_test == "SocialGNN_V":
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'NUM_NODES NUM_AGENTS V_SPATIAL_SIZE E_SPATIAL_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  C = model_config(NUM_NODES = 4, NUM_AGENTS = 4, V_SPATIAL_SIZE = 12, E_SPATIAL_SIZE = 12, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  model = SocialGNN(Videos, C, sample_graph_dicts_list)
  model._initialize_session()
  model.load_model(string + model_to_test)

  accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)

  inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
  TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
  PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
  
  with open('./Outputs/Predictions/' + string[16:] + model_to_test, "wb") as f:
    pickle.dump(TL, f)
    pickle.dump(PL, f)

elif model_to_test == "SocialGNN_E":
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'NUM_NODES MAX_EDGES E_SPATIAL_SIZE E_TEMPORAL_SIZE E_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 12, E_TEMPORAL_SIZE = 6, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  N_EPOCHS = 150
  model = SocialGNN_E(Videos, C, sample_graph_dicts_list)
  model._initialize_session()
  model.load_model(string + model_to_test)

  accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)

  inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
  TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
  PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
  
  with open('./Outputs/Predictions/' + string[16:] + model_to_test, "wb") as f:
    pickle.dump(TL, f)
    pickle.dump(PL, f)

elif model_to_test == "CueBasedLSTM":
  #CueBasedLSTM-Relation: set 28-->40 explicit_edges=True, and output file name append -Relation
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

  C = model_config(FEATURE_SIZE = 28, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  model = CueBasedLSTM(Videos, C, explicit_edges = False)
  model._initialize_session()
  model.load_model(string + model_to_test)

  accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)

  inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
  TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
  PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
  
  with open('./Outputs/Predictions/' + string[16:] + model_to_test, "wb") as f:
    pickle.dump(TL, f)
    pickle.dump(PL, f)




    
