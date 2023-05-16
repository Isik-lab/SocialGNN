# Get final layer activations during prediction for Models (SocialGNN / CueBasedLSTM)
from collections import Counter

import os, sys
import pickle
import numpy as np
from collections import namedtuple
from datetime import datetime

import time
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help="SocialGNN_V/SocialGNN_E/CueBasedLSTM/CueBasedLSTM-Relation", type= str)
parser.add_argument('--context_info', help="True/False", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument('--bootstrap_no', help="0-9", type=int, default=0)
parser.add_argument('--dataset', help="main_set/generalization_set/middle10s_set", type= str, default = "main_set")
parser.add_argument('--train_datetime', help="YYYYMMDD", type= str, default= "20230503")
parser.add_argument('--activation_type', help="classifier/RNN", type= str)

args = parser.parse_args()
timestr = time.strftime("%Y%m%d")


if args.dataset == "main_set":
  #from SocialGNN_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline
  from SocialGNN_withU_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline

  ### LOAD DATA
  #model_string = './TrainedModels/PHASE_mysplit_humanratings_May1_' + str(args.bootstrap_no) + '_'
  model_string = './TrainedModels/PHASE_mysplit_humanratings_context'+ str(args.context_info) + '_' + str(args.train_datetime) + '_' + str(args.bootstrap_no) + '_'
  
  testfile = './PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_' + str(args.bootstrap_no)

  with open('./PHASE/Videos_humanratings', "rb") as f:
    Videos = pickle.load(f)

  with open(testfile, "rb") as f:
    X_train = pickle.load(f)
    y_train = pickle.load(f)
    X_test = pickle.load(f)
    y_test = pickle.load(f) 

  print(Counter(y_train), Counter(y_test))

elif args.dataset == "generalization_set":
  from SocialGNN_withU_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline

  #model_string = './TrainedModels/PHASE_originalsplit_withcontext_June28_'
  model_string = './TrainedModels/PHASE_originalsplit_context'+ str(args.context_info) + '_' + str(args.train_datetime) + '_'

  with open('./PHASE/Videos_humanratings', "rb") as f:
    Videos = pickle.load(f)
  with open('./PHASE/Videos_Test_humanratings', "rb") as f:
    Test_Videos = pickle.load(f)

    n_train_vid = len(Videos)
    n_test_vid = len(Test_Videos)
    print(len(Videos), len(Test_Videos))
    Videos.extend(Test_Videos)

    ### SPLIT THE DATA
    X_train = range(n_train_vid)
    X_test = range(n_train_vid,n_train_vid+n_test_vid)
    y_train = [Videos[v]['social_goals'] for v in X_train]
    y_test = [Videos[v]['social_goals'] for v in X_test]

    print(Counter(y_train), Counter(y_test))

elif args.dataset == "middle10s_set":
  from SocialGNN_withU_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline

  model_string = './TrainedModels/PHASE_originalsplit_withcontext_June28_'

  with open('./PHASE/Videos_humanratings', "rb") as f:
    Videos = pickle.load(f)
  with open('../SI_Comp_fMRI/test_media_middle10s/Videos_test_middle10s', "rb") as f:
    Test_Videos = pickle.load(f)

    n_train_vid = len(Videos)
    n_test_vid = len(Test_Videos)
    print(len(Videos), len(Test_Videos))
    Videos.extend(Test_Videos)

    ### SPLIT THE DATA
    X_train = range(n_train_vid)
    X_test = range(n_train_vid,n_train_vid+n_test_vid)
    y_train = [Videos[v]['social_goals'] for v in X_train]
    y_test = [Videos[v]['social_goals'] for v in X_test]

    print(Counter(y_train), Counter(y_test))

### SPLIT THE DATA
mapping = {'friendly':(1,0,0), 'neutral': (0,1,0), 'adversarial': (0,0,1)}


if args.model_name == "SocialGNN_E":
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'NUM_NODES MAX_EDGES E_SPATIAL_SIZE E_TEMPORAL_SIZE E_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  if args.dataset == "main_set":
    C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 64, E_TEMPORAL_SIZE = 16, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  elif args.dataset == "generalization_set" or args.dataset == "middle10s_set":
    C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 64, E_TEMPORAL_SIZE = 16, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
    
  model = SocialGNN_E(Videos, C, args.context_info, sample_graph_dicts_list)
  model._initialize_session()
  model.load_model(model_string + args.model_name)

  activations_values = model.get_activations(test_data_idx=X_test, mapping=mapping, type=args.activation_type)
  activations = {Videos[X_test[i]]['name']:activations_values[i] for i in range(len(activations_values))}
  print(len(activations), activations_values[0])

  accuracy = model.test(test_data_idx=X_test, mapping=mapping)
  print("Test Acc:",accuracy)

  with open('./Outputs/Activations/' + args.activation_type + '_activations_' + model_string[16:] + args.model_name + '_' + args.dataset + '_' + datetime.now().strftime("%d-%m-%Y"), "wb") as f:
        pickle.dump(activations, f)

elif args.model_name == "CueBasedLSTM" or args.model_name == "CueBasedLSTM-Relation":
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

  if args.model_name == "CueBasedLSTM":
    f_size = 52 if args.context_info==True else 28
    e = False
  else:
    f_size = 64 if args.context_info==True else 40
    e = True

  if args.dataset == "main_set":
    C = model_config(FEATURE_SIZE = f_size, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  elif args.dataset == "generalization_set":
    C = model_config(FEATURE_SIZE = f_size, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
  
  model = CueBasedLSTM(Videos, C, args.context_info, explicit_edges = e)
  model._initialize_session()
  model.load_model(model_string + args.model_name)

  activations_values = model.get_activations(test_data_idx=X_test, mapping=mapping, type=args.activation_type)
  activations = {Videos[X_test[i]]['name']:activations_values[i] for i in range(len(activations_values))}
  print(len(activations), activations_values[0])

  accuracy = model.test(test_data_idx=X_test, mapping=mapping)
  print("Test Acc:",accuracy)

  with open('./Outputs/Activations/' + args.activation_type + '_activations_' + model_string[16:] + args.model_name + '_' + args.dataset + '_' + datetime.now().strftime("%d-%m-%Y"), "wb") as f:
        pickle.dump(activations, f)