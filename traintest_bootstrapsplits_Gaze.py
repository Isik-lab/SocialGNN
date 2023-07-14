from SocialGNN_modelfunctions_Gaze import SocialGNN, get_inputs_outputs, CueBasedLSTM, get_inputs_outputs_baseline
from collections import namedtuple, Counter
import pickle
import numpy as np
import sys

mode = sys.argv[1]
which_model = sys.argv[2]
output_labels_type = int(sys.argv[3]) #2 for social/non social, 5 for gaze labels
dataset = sys.argv[4] #5Jun23 or 14Jun23
#dataset = "5Jun23"
string = './TrainedModels/GazeDataset_Jun1523_traintest' + dataset + '_'
bootstrap_splits = 10

confusion_matrices = []
# Train using model
if which_model == 'SocialGNN_V':
	model_config = namedtuple('model_config', 'NUM_NODES NUM_AGENTS V_SPATIAL_SIZE E_SPATIAL_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

	scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

	for split in range(bootstrap_splits):
		print("\n\n##### Bootstrap Split No.:", split)
		with open('./Gaze_Dataset/bootstrapped_traintest_splits_pickles/traintest_seqs_'+dataset+'_'+str(split), 'rb') as f:
			Sequences = pickle.load(f)
			seq_train_idx = pickle.load(f)
			seq_test_idx = pickle.load(f)

		labels = [seq['label'][0] for seq in Sequences]

		if output_labels_type == 2:
			mapping = {'AvertGaze':(1,0), 'GazeFollow': (1,0), 'JointAtt': (1,0), 'MutualGaze': (1,0), 'SingleGaze' : (0,1)}
			C = model_config(NUM_NODES = 5, NUM_AGENTS = 5, V_SPATIAL_SIZE = 12, E_SPATIAL_SIZE = 12, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 2, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,1.5]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
			map_2_social_nonsocial = {'AvertGaze':'Social', 'GazeFollow': 'Social', 'JointAtt': 'Social', 'MutualGaze': 'Social', 'SingleGaze' : 'NonSocial'}
			labels_for_skf = [map_2_social_nonsocial[x] for x in labels] 	#binary labels
		else:
			mapping = {'AvertGaze':(1,0,0,0,0), 'GazeFollow': (0,1,0,0,0), 'JointAtt': (0,0,1,0,0), 'MutualGaze': (0,0,0,1,0), 'SingleGaze' : (0,0,0,0,1)}
			C = model_config(NUM_NODES = 5, NUM_AGENTS = 5, V_SPATIAL_SIZE = 12, E_SPATIAL_SIZE = 12, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 5, BATCH_SIZE = 20, CLASS_WEIGHTS = [[5.69,4.42,1.85,1.66,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
			labels_for_skf = labels

		sample_graph_dicts_list, _, _ = get_inputs_outputs(Sequences[:20])
		N_EPOCHS = 150
		model = SocialGNN(Sequences, C, sample_graph_dicts_list)
		model._initialize_session()
		
		if mode == "train":
			scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train=seq_train_idx, y_train=np.array(labels_for_skf)[seq_train_idx], mapping=mapping))
			scores['entire_trainset'].append(model.test(test_data_idx = seq_train_idx, mapping=mapping))
			scores['testset'].append(model.test(test_data_idx = seq_test_idx, mapping=mapping))
			print(scores)
			model.save_model(string + str(output_labels_type) + '_' + str(split) + '_' + which_model)
		else:
			model.load_model(string + str(output_labels_type) + '_' + str(split) + '_' + which_model)
			scores['testset'].append(model.test(test_data_idx = seq_test_idx, mapping=mapping))
			print(scores)

			import sklearn
			accuracy, true_labels, pred_labels = model.test(test_data_idx=seq_test_idx, mapping=mapping, output_predictions = True)
			if split==0:
				confusion_matrices = sklearn.metrics.confusion_matrix(true_labels,pred_labels)
			else:
				confusion_matrices = np.dstack((confusion_matrices, sklearn.metrics.confusion_matrix(true_labels,pred_labels)))

	print(np.mean(confusion_matrices, axis=2))
	print(np.sum(confusion_matrices, axis=2))

elif which_model == 'CueBasedLSTM':
	model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

	scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

	for split in range(bootstrap_splits):
		print("\n\n##### Bootstrap Split No.:", split)
		with open('./Gaze_Dataset/bootstrapped_traintest_splits_pickles/traintest_seqs_'+dataset+'_'+str(split), 'rb') as f:
			Sequences = pickle.load(f)
			seq_train_idx = pickle.load(f)
			seq_test_idx = pickle.load(f)

		labels = [seq['label'][0] for seq in Sequences]

		if output_labels_type == 2:
			mapping = {'AvertGaze':(1,0), 'GazeFollow': (1,0), 'JointAtt': (1,0), 'MutualGaze': (1,0), 'SingleGaze' : (0,1)}
			C = model_config(FEATURE_SIZE = 125, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 2, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,1.5]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
			map_2_social_nonsocial = {'AvertGaze':'Social', 'GazeFollow': 'Social', 'JointAtt': 'Social', 'MutualGaze': 'Social', 'SingleGaze' : 'NonSocial'}
			labels_for_skf = [map_2_social_nonsocial[x] for x in labels] 	#binary labels
		elif output_labels_type ==5:
			mapping = {'AvertGaze':(1,0,0,0,0), 'GazeFollow': (0,1,0,0,0), 'JointAtt': (0,0,1,0,0), 'MutualGaze': (0,0,0,1,0), 'SingleGaze' : (0,0,0,0,1)}
			C = model_config(FEATURE_SIZE = 125, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 5, BATCH_SIZE = 20, CLASS_WEIGHTS = [[5.69,4.42,1.85,1.66,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
			labels_for_skf = labels

		N_EPOCHS = 150
		model = CueBasedLSTM(Sequences, C, explicit_edges=False)
		model._initialize_session()

		if mode == "train":
			scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train=seq_train_idx, y_train=np.array(labels_for_skf)[seq_train_idx], mapping=mapping))
			scores['entire_trainset'].append(model.test(test_data_idx = seq_train_idx, mapping=mapping))
			scores['testset'].append(model.test(test_data_idx = seq_test_idx, mapping=mapping))
			print(scores)
			model.save_model(string + str(output_labels_type) + '_' + str(split) + '_' + which_model)
		else:
			model.load_model(string + str(output_labels_type) + '_' + str(split) + '_' + which_model)
			scores['testset'].append(model.test(test_data_idx = seq_test_idx, mapping=mapping))
			print(scores)

			import sklearn
			accuracy, true_labels, pred_labels = model.test(test_data_idx=seq_test_idx, mapping=mapping, output_predictions = True)
			print(Counter(true_labels))
			if split==0:
				confusion_matrices = sklearn.metrics.confusion_matrix(true_labels,pred_labels)
			else:
				confusion_matrices = np.dstack((confusion_matrices, sklearn.metrics.confusion_matrix(true_labels,pred_labels)))

	print(np.mean(confusion_matrices, axis=2))
	print(np.sum(confusion_matrices, axis=2))

elif which_model == 'CueBasedLSTM-Relation':
	model_config = namedtuple('model_config', 'FEATURE_SIZE V_TEMPORAL_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

	scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

	for split in range(bootstrap_splits):
		print("\n\n##### Bootstrap Split No.:", split)
		with open('./Gaze_Dataset/bootstrapped_traintest_splits_pickles/traintest_seqs_'+dataset+'_'+str(split), 'rb') as f:
			Sequences = pickle.load(f)
			seq_train_idx = pickle.load(f)
			seq_test_idx = pickle.load(f)

		labels = [seq['label'][0] for seq in Sequences]

		if output_labels_type == 2:
			mapping = {'AvertGaze':(1,0), 'GazeFollow': (1,0), 'JointAtt': (1,0), 'MutualGaze': (1,0), 'SingleGaze' : (0,1)}
			C = model_config(FEATURE_SIZE = 145, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 2, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,1.5]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
			map_2_social_nonsocial = {'AvertGaze':'Social', 'GazeFollow': 'Social', 'JointAtt': 'Social', 'MutualGaze': 'Social', 'SingleGaze' : 'NonSocial'}
			labels_for_skf = [map_2_social_nonsocial[x] for x in labels] 	#binary labels
		elif output_labels_type ==5:
			mapping = {'AvertGaze':(1,0,0,0,0), 'GazeFollow': (0,1,0,0,0), 'JointAtt': (0,0,1,0,0), 'MutualGaze': (0,0,0,1,0), 'SingleGaze' : (0,0,0,0,1)}
			C = model_config(FEATURE_SIZE = 145, V_TEMPORAL_SIZE = 6, V_OUTPUT_SIZE = 5, BATCH_SIZE = 20, CLASS_WEIGHTS = [[5.69,4.42,1.85,1.66,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
			labels_for_skf = labels

		N_EPOCHS = 150
		model = CueBasedLSTM(Sequences, C, explicit_edges=True)
		model._initialize_session()

		if mode == "train":
			scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train=seq_train_idx, y_train=np.array(labels_for_skf)[seq_train_idx], mapping=mapping))
			scores['entire_trainset'].append(model.test(test_data_idx = seq_train_idx, mapping=mapping))
			scores['testset'].append(model.test(test_data_idx = seq_test_idx, mapping=mapping))
			print(scores)
			model.save_model(string + str(output_labels_type) + '_' + str(split) + '_' + which_model)
		else:
			model.load_model(string + str(output_labels_type) + '_' + str(split) + '_' + which_model)
			scores['testset'].append(model.test(test_data_idx = seq_test_idx, mapping=mapping))
			print(scores)

			import sklearn
			accuracy, true_labels, pred_labels = model.test(test_data_idx=seq_test_idx, mapping=mapping, output_predictions = True)
			print(Counter(true_labels))
			if split==0:
				confusion_matrices = sklearn.metrics.confusion_matrix(true_labels,pred_labels)
			else:
				confusion_matrices = np.dstack((confusion_matrices, sklearn.metrics.confusion_matrix(true_labels,pred_labels)))


	print(np.sum(confusion_matrices, axis=2))
