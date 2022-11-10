from SocialGNN_withU_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline
import os, sys
import pickle
import numpy as np
from collections import namedtuple, Counter
from sklearn.model_selection import ParameterGrid

model_to_train = sys.argv[1]
string = './TrainedModels/PHASE_originalsplit_withcontext_June28_'
mode = "test"

### LOAD TRAIN DATA
## Get videos data (position, vel, landmark info etc and associated human rating labels)
with open('./PHASE/Videos_humanratings', "rb") as f:
  Videos = pickle.load(f)

### LOAD TEST DATA
with open('./PHASE/Videos_Test_humanratings', "rb") as f:
  Test_Videos = pickle.load(f)


n_train_vid = len(Videos)
n_test_vid = len(Test_Videos)
Videos.extend(Test_Videos)

X_train = range(n_train_vid)
X_test = range(n_train_vid,n_train_vid+n_test_vid)
y_train = [Videos[v]['social_goals'] for v in X_train]
y_test = [Videos[v]['social_goals'] for v in X_test]

mapping = {'friendly':(1,0,0), 'neutral': (0,1,0), 'adversarial': (0,0,1)}


if mode == "hyperparameter_tune":

  def get_model_config(model_config, specific_params = None, model_to_tune = "SocialGNN_V"):
    if model_to_tune == "SocialGNN_V":
      C = model_config(NUM_NODES = 4, NUM_AGENTS = 4, V_SPATIAL_SIZE = 12, E_SPATIAL_SIZE = 12, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
    elif model_to_tune == "SocialGNN_E":
      C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 12, E_TEMPORAL_SIZE = 6, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 5e-3, LAMBDA = 0.01 )
    elif model_to_tune == "CueBasedLSTM":
      C = model_config(FEATURE_SIZE = 28, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = 
  [[1.0,2.0,1.0]], LEARNING_RATE = 5e-3, LAMBDA = 0.01 )

    if specific_params == None:
      return C
    for k,v in specific_params.items():
      if k == "V_param":
        C = C._replace(V_SPATIAL_SIZE = v[0])
        C = C._replace(V_TEMPORAL_SIZE = v[1])
        C = C._replace(E_SPATIAL_SIZE = v[0])
      if k == "E_param":
        C = C._replace(E_SPATIAL_SIZE = v[0])
        C = C._replace(E_TEMPORAL_SIZE = v[1])
      if k == "lr_param":
        C = C._replace(LEARNING_RATE = v)
      if k == "reg_param":
        C = C._replace(LAMBDA = v)
      if k == "LSTM_size":
        C = C._replace(V_TEMPORAL_SIZE = v)
    return C

  if model_to_train == "SocialGNN_V":
    print("\n.............HYPER-PARAMETER TUNING..............")
    param = {"V_param" : [(12,6), (64,16)], "lr_param": [1e-3, 5e-3, 1e-2, 5e-2], "reg_param": [0.01, 0.05, 0.1]}

    model_config = namedtuple('model_config', 'NUM_NODES NUM_AGENTS V_SPATIAL_SIZE E_SPATIAL_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
    sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

    C_paramgrid = ParameterGrid(param)
    scores = {'cross_val':[], 'entire_trainset':[]}

    for combination in C_paramgrid:
      print("\n\n",combination)
      C = get_model_config(model_config, combination)
      N_EPOCHS = 150
      model = SocialGNN(Videos, C, sample_graph_dicts_list)
      model._initialize_session()
      scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
      scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
      print(scores)

    for i,combination in enumerate(C_paramgrid):
      print(combination, scores['cross_val'][i], scores['entire_trainset'][i])

  elif model_to_train == "CueBasedLSTM":
    print("\n.............HYPER-PARAMETER TUNING..............")
    param = {"LSTM_size" : [6, 16, 64], "lr_param": [1e-3, 5e-3, 1e-2, 5e-2], "reg_param": [0.01, 0.05, 0.1]}

    model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

    C_paramgrid = ParameterGrid(param)
    scores = {'cross_val':[], 'entire_trainset':[]}

    for combination in C_paramgrid:
      print("\n\n",combination)
      C = get_model_config(model_config, combination, model_to_tune = "CueBasedLSTM")
      N_EPOCHS = 150
      model = CueBasedLSTM(Videos, C)
      model._initialize_session()
      scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
      scores['entire_trainset'].append(model.test(test_data_idx=X_train, mapping=mapping))
      print(scores)

    for i,combination in enumerate(C_paramgrid):
      print(combination, scores['cross_val'][i], scores['entire_trainset'][i])

elif mode == "train" or mode == "test":

  if model_to_train == "SocialGNN_V":
    model_config = namedtuple('model_config', 'NUM_NODES NUM_AGENTS V_SPATIAL_SIZE E_SPATIAL_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
    sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

    C = model_config(NUM_NODES = 4, NUM_AGENTS = 4, V_SPATIAL_SIZE = 64, E_SPATIAL_SIZE = 64, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 150
    model = SocialGNN(Videos, C, sample_graph_dicts_list)
    model._initialize_session()
    
    if mode == "train":
      print("Cross Val Score:",model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
      print("Train Acc:",model.test(test_data_idx=X_train, mapping=mapping))
      print("Test Acc:",model.test(test_data_idx=X_test, mapping=mapping))

      model.save_model(string + model_to_train)
    else:
      model.load_model(string + model_to_train)
      accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)
      print("Test Acc:",accuracy)

      inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
      TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
      PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
      
      with open('./Outputs/Predictions/' + string[16:] + model_to_train, "wb") as f:
        pickle.dump(TL, f)
        pickle.dump(PL, f)

    
  elif model_to_train == "SocialGNN_E":
    model_config = namedtuple('model_config', 'NUM_NODES MAX_EDGES E_SPATIAL_SIZE E_TEMPORAL_SIZE E_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
    sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

    C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 64, E_TEMPORAL_SIZE = 16, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    N_EPOCHS = 150
    model = SocialGNN_E(Videos, C, sample_graph_dicts_list)
    model._initialize_session()
    
    if mode == "train":
      print("Cross Val Score:",model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
      print("Train Acc:",model.test(test_data_idx=X_train, mapping=mapping))
      print("Test Acc:",model.test(test_data_idx=X_test, mapping=mapping))

      model.save_model(string + model_to_train)
    else:
      model.load_model(string + model_to_train)
      accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)
      print("Test Acc:",accuracy)
      
      inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
      TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
      PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
      
      with open('./Outputs/Predictions/' + string[16:] + model_to_train, "wb") as f:
        pickle.dump(TL, f)
        pickle.dump(PL, f)
    
  elif model_to_train == "CueBasedLSTM":
    #CueBasedLSTM-Relation: set 28-->40 explicit_edges=True, and output file name append -Relation
    model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

    C = model_config(FEATURE_SIZE = 52, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
    N_EPOCHS = 150
    model = CueBasedLSTM(Videos, C)
    model._initialize_session()
    
    if mode == "train":
      print("Cross Val Score:",model.cross_validate(5,N_EPOCHS, X_train, y_train, mapping))
      print("Train Acc:",model.test(test_data_idx=X_train, mapping=mapping))
      print("Test Acc:",model.test(test_data_idx=X_test, mapping=mapping))

      model.save_model(string + model_to_train)
    else:
      model.load_model(string + model_to_train)
      accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)
      print("Test Acc:",accuracy)

      inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
      TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
      PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
      
      with open('./Outputs/Predictions/' + string[16:] + model_to_train, "wb") as f:
        pickle.dump(TL, f)
        pickle.dump(PL, f)

