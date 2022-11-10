import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

### Load and combine all processed dictionaries
V = {}
with open('./preprocessed_pickles/processed_Mar25_0', 'rb') as f:
	V.update(pickle.load(f))

with open('./preprocessed_pickles/processed_Mar25_100', 'rb') as f:
	V.update(pickle.load(f))

with open('./preprocessed_pickles/processed_Mar25_200', 'rb') as f:
	V.update(pickle.load(f))

bootstrapping = 10

if bootstrapping == False:
	string = './bootstrapped_traintest_splits_pickles/traintest_seqs_29Jul22_notbootstrapped'
	bootstrapping = 1
else:
	string = './bootstrapped_traintest_splits_pickles/traintest_seqs_29Jul22_'

# format: V[video_id]['graph_dicts' or 'sequences' or 'labels'][sequence no][entry in seq or graph dict for frame]

def check_label_anomalies(seq_labels):
	counts = Counter(seq_labels)
	if 'JointAtt' in counts.keys() and counts['JointAtt'] == 1:
		return False
	elif 'MutualGaze' in counts.keys() and counts['MutualGaze'] == 1:
		return False
	elif 'NA' in counts.keys() and counts['NA'] >= 1:
		return False
	else:
		return True

for i in range(bootstrapping):
	### Split Videos into train and test (stratify not possible because will have to separate labels then)
	V_train_idx, V_test_idx = train_test_split(list(V.keys()))
	print("Train Videos", len(V_train_idx), "Test Videos", len(V_test_idx))

	Sequences = []
	seq_train_idx = []
	seq_test_idx = []

	for k in V_train_idx:
		seq_labels = [tuple(v for k,v in seq.items() if k[0]=='P') for seq in V[k]['labels']]
		for ind, seq in enumerate(seq_labels):
			if check_label_anomalies(seq) and len(seq)>=2 and len(V[k]['labels'][ind])<=5:  #remove this last condition later
				Seq_dict = dict()
				Seq_dict['label'] = seq[::-1]  # reversing RETHINK THIS

				# reverse node features (keeping last two persons ahead) ##RETHINK THIS
				for frame_id in range(len(V[k]['graph_dicts'][ind])):
					node_features = np.array(V[k]['graph_dicts'][ind][frame_id]['nodes'])
					V[k]['graph_dicts'][ind][frame_id]['nodes'] = node_features[::-1].tolist()

					padding = np.zeros((5-len(node_features),node_features.shape[1])).tolist()
					V[k]['graph_dicts'][ind][frame_id]['nodes'].extend(padding)

					if 'problematic' in V[k]['graph_dicts'][ind][frame_id]['senders']:
						print("senders", V[k]['graph_dicts'][ind][frame_id]['senders'])

				Seq_dict['graph_dicts'] = V[k]['graph_dicts'][ind]

				seq_train_idx.append(len(Sequences))
				Sequences.append(Seq_dict)

	for k in V_test_idx:
		seq_labels = [tuple(v for k,v in seq.items() if k[0]=='P') for seq in V[k]['labels']]
		for ind, seq in enumerate(seq_labels):
			if check_label_anomalies(seq) and len(seq)>=2 and len(V[k]['labels'][ind])<=5:  #remove this last condition later
				Seq_dict = dict()
				Seq_dict['label'] = seq[::-1]  # reversing RETHINK THIS

				# reverse node features (keeping last two persons ahead) ##RETHINK THIS
				for frame_id in range(len(V[k]['graph_dicts'][ind])):
					node_features = np.array(V[k]['graph_dicts'][ind][frame_id]['nodes'])
					V[k]['graph_dicts'][ind][frame_id]['nodes'] = node_features[::-1].tolist()

					padding = np.zeros((5-len(node_features),node_features.shape[1])).tolist()
					V[k]['graph_dicts'][ind][frame_id]['nodes'].extend(padding)

				Seq_dict['graph_dicts'] = V[k]['graph_dicts'][ind]

				seq_test_idx.append(len(Sequences))
				Sequences.append(Seq_dict)

	print("Train Seqs", len(seq_train_idx), "Test Seqs", len(seq_test_idx))

	s = string + str(i)
	with open(s, 'wb') as f:
		pickle.dump(Sequences, f)
		pickle.dump(seq_train_idx, f)
		pickle.dump(seq_test_idx, f)
