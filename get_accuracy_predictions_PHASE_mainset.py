#from SocialGNN_modelfunctions import SocialGNN, SocialGNN_E, CueBasedLSTM, get_inputs_outputs, get_inputs_outputs_baseline
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
parser.add_argument('--train_datetime', help="YYYYMMDD", type= str, default= "20230503")
parser.add_argument('--context_info', help="True/False", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument('--bootstrap_no', help="0-9", type=int, default=0)
parser.add_argument('--save_predictions', help="True/False", type=lambda x: bool(strtobool(x)), default = True)

args = parser.parse_args()
timestr = time.strftime("%Y%m%d")

model_string = './TrainedModels/PHASE_mysplit_humanratings_context'+ str(args.context_info) + '_' + str(args.train_datetime) + '_'
bootstrap_file = './PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_' + str(args.bootstrap_no)

'''
### LOAD DATA
bootstrap_no = sys.argv[1]
model_to_test = sys.argv[2]
string = './TrainedModels/PHASE_mysplit_humanratings_May1_' + bootstrap_no + '_'
bootstrap_file = './PHASE/bootstrapped_traintest_splits_pickles/XY_Apr30_humanratings_' + bootstrap_no
'''

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

if args.model_name == "SocialGNN_V" or args.model_name == "SocialGNN_V_onlyagents":
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'NUM_NODES NUM_AGENTS V_SPATIAL_SIZE E_SPATIAL_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  a = 2 if args.model_name == "SocialGNN_V_onlyagents" else 4

  C = model_config(NUM_NODES = 4, NUM_AGENTS = a, V_SPATIAL_SIZE = 64, E_SPATIAL_SIZE = 64, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  model = SocialGNN(Videos, C, args.context_info, sample_graph_dicts_list)
  model._initialize_session()
  model.load_model(model_string  + str(args.bootstrap_no) + '_' + args.model_name)

  accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = args.save_predictions)

  inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
  TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
  PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
  
  with open('./Outputs/Predictions/' + model_string[16:] + '_' + timestr + '_' + str(args.bootstrap_no) + '_' + args.model_name, "wb") as f:
      pickle.dump(TL, f)
      pickle.dump(PL, f)

elif args.model_name == "SocialGNN_E" or args.model_name == "SocialGNN_E_onlyagentedges":
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'NUM_NODES MAX_EDGES E_SPATIAL_SIZE E_TEMPORAL_SIZE E_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')
  sample_graph_dicts_list, _, _ = get_inputs_outputs(Videos[:20])

  a = True if args.model_name == "SocialGNN_E_onlyagentedges" else False

  C = model_config(NUM_NODES = 4, MAX_EDGES = 12, E_SPATIAL_SIZE = 64, E_TEMPORAL_SIZE = 16, E_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  N_EPOCHS = 150
  model = SocialGNN_E(Videos, C, args.context_info, sample_graph_dicts_list, ablate = a)
  model._initialize_session()
  model.load_model(model_string  + str(args.bootstrap_no) + '_' + args.model_name)

  accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)

  inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
  TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
  PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
  
  with open('./Outputs/Predictions/' + model_string[16:] + '_' + timestr + '_' + str(args.bootstrap_no) + '_' + args.model_name, "wb") as f:
      pickle.dump(TL, f)
      pickle.dump(PL, f)

elif args.model_name == "CueBasedLSTM" or args.model_name == "CueBasedLSTM-Relation":
  #CueBasedLSTM-Relation: set 28-->40 explicit_edges=True, and output file name append -Relation
  print("\n.............TESTING..............")

  model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

  if args.model_name == "CueBasedLSTM":
    f_size = 52 if args.context_info==True else 28
    e = False
  else:
    f_size = 64 if args.context_info==True else 40
    e = True

  C = model_config(FEATURE_SIZE = f_size, V_TEMPORAL_SIZE = 16, V_OUTPUT_SIZE = 3, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,2.0,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.05 )
  model = CueBasedLSTM(Videos, C, args.context_info, explicit_edges = e)
  model._initialize_session()
  model.load_model(model_string  + str(args.bootstrap_no) + '_' + args.model_name)

  accuracy, true_labels, pred_labels = model.test(test_data_idx=X_test, mapping=mapping, output_predictions = True)

  inv_mapping = {0:'friendly',  1:'neutral',  2:'adversarial'}
  TL = {Videos[X_test[i]]['name']:inv_mapping[true_labels[i]] for i in range(len(true_labels))}
  PL = {Videos[X_test[i]]['name']:inv_mapping[pred_labels[i]] for i in range(len(pred_labels))}
  
  with open('./Outputs/Predictions/' + model_string[16:] + '_' + timestr + '_' + str(args.bootstrap_no) + '_' + args.model_name, "wb") as f:
      pickle.dump(TL, f)
      pickle.dump(PL, f)




    
