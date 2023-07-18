import numpy as np
import itertools
import tensorflow as tf
import sonnet as snt
from collections import namedtuple, Counter
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import sklearn
import sys
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="train/test", type= str)
parser.add_argument('--dataset', help="5Jun23/14Jun23", type= str)
parser.add_argument('--output_type', help="2 for social/non social, 5 for gaze labels", type= int)

args = parser.parse_args()

which_model = "vgg19_full"
string = './TrainedModels/GazeDataset_Jul30_traintest' + args.dataset + '_'
bootstrap_splits = 10

def get_inputs_outputs_vgg19full(Sequences, explicit_edges = False):
  seq_features_list = []
  labels_social = []
  video_timesteps = []

  for seq in Sequences:
    #curr_seq_features = [np.concatenate(frame['nodes']).tolist() for frame in seq['graph_dicts']]
    curr_seq_features = []
    for frame in seq['graph_dicts']:
      nodes = frame['nodes']

      if explicit_edges:
        senders = frame['senders']
        receivers = frame['receivers']

        # all possible edges
        x = list(itertools.combinations(range(5), 2))
        x.extend([(rx, sx) for sx,rx in x])
        d = {key:0 for key in x}
        #fill in actual edges
        for edge in range(len(senders)):
          d[(senders[edge],receivers[edge])] = 1
        edges = [float(x) for x in d.values()]
        # add it to feature vector
        nodes.extend(edges)

      curr_seq_features.append(nodes)

    curr_seq_features = np.array(curr_seq_features)
    avg_seq_features = np.mean(curr_seq_features, axis =0)

    seq_features_list.append(avg_seq_features)
    labels_social.append(seq['label'])

  return seq_features_list, labels_social

ground_truth = "given_labels"

class VGG19_wClassifier(object):
  def __init__(self, dataset, config, explicit_edges=False):
    self.graph = tf.Graph()

    self.dataset = dataset
    self.config = config  #define model parameters
    self.explicit_edges = explicit_edges #whether it has explicit relationship/edge info

    with self.graph.as_default():
      self._define_inputs()
      self._build_graph()
      self.initializer = tf.global_variables_initializer()
      self.saver = tf.train.Saver()
    self._initialize_session()
  
  def _initialize_session(self):
    print("\n.............INITIALIZATION SESSION..............")
    try:
      sess.close()
    except NameError:
      pass
    self.sess = tf.Session(graph=self.graph)
    self.sess.run(self.initializer)

  def _define_inputs(self):
    print(".............DEFINING INPUT PLACEHOLDERS..............")
    self.X = tf.placeholder(tf.float32,shape=[None, self.config.FEATURE_SIZE])
    self.target_V_placeholder = tf.placeholder(tf.float32,shape=[None, self.config.V_OUTPUT_SIZE])

  def _build_graph(self):
    print("\n.............BUILDING GRAPH..............")
    #########   Define Layers/Blocks    #########
    classifier = snt.Linear(self.config.V_OUTPUT_SIZE)
    
    output_label_V = classifier(self.X)  
    self.output_label_V = output_label_V

    #########   Training Loss + Optimizer    #########
    weights = tf.reduce_sum(self.config.CLASS_WEIGHTS * self.target_V_placeholder, axis=1)  # deduce weights for batch samples based on their true label
    self.loss_V = tf.losses.softmax_cross_entropy(self.target_V_placeholder, output_label_V, weights=weights)
    self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.config.LAMBDA
    self.loss = self.loss_V + self.lossL2 
    self.optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
    self.step_op = self.optimizer.minimize(self.loss)
    #print("\nLoss", self.loss_V, self.loss)
    #print("Optimizer",self.optimizer)
    print("\nTrainable paramters: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


  def train(self, N_EPOCHS, train_data_idx, mapping, plot=False):
    print("\n.............TRAINING..............")
    training_loss = {'total_loss':[], 'loss_V':[]}
    for e in range(N_EPOCHS):
      np.random.shuffle(train_data_idx)
      batches = [train_data_idx[k:k+self.config.BATCH_SIZE] for k in range(0, len(train_data_idx), self.config.BATCH_SIZE)]
      
      epoch_loss = 0
      epoch_loss_V = 0
      for batch in batches:
        if len(batch)<self.config.BATCH_SIZE:
          # pad if batch smaller than BATCH_SIZE
          for x in range(self.config.BATCH_SIZE - len(batch)):
            batch = np.append(batch, batch[0])

        # get X_train and Y_train from dataset + train_idx
        X, input_labels_social = get_inputs_outputs_vgg19full(np.array(self.dataset)[batch], self.explicit_edges)
        if ground_truth == 'human_ratings':
          input_labels = [mapping[x] for x in input_labels_social]
        else:
          input_labels = [mapping[x[0]] for x in input_labels_social] #only agent 0

        # feed dictionary
        feed_dict = {}
        feed_dict[self.X] = X
        feed_dict[self.target_V_placeholder] = input_labels

        #print(feed_dict)
        # train
        train_values = self.sess.run({"step": self.step_op,
          "loss": self.loss, "loss_V": self.loss_V, "output_label_V": self.output_label_V}, feed_dict)
        
        epoch_loss += train_values['loss']
        epoch_loss_V += train_values['loss_V']

      print("Epoch No.:",e,"\tLoss:",epoch_loss/len(batches), "\tLoss_V:",epoch_loss_V/len(batches))
      training_loss['total_loss'].append(epoch_loss/len(batches))
      training_loss['loss_V'].append(epoch_loss_V/len(batches))

    if plot==True:
      plt.figure(figsize=(10,7))
      plt.plot(training_loss['total_loss'], label='total loss')
      plt.plot(training_loss['loss_V'], label = 'loss_V')
      plt.legend()
      plt.ylabel('Training Loss')
      plt.xlabel('Number of Epochs')
      plt.show()

  def test(self, test_data_idx, mapping, output_predictions=False):
    print("\n.............TESTING..............")
    test_loss = 0
    test_loss_V = 0
    accuracy_batch = []
    V0_pred = []
    V0_true = []

    batches = [test_data_idx[k:k+self.config.BATCH_SIZE] for k in range(0, len(test_data_idx), self.config.BATCH_SIZE)]
    for batch in batches:
      # pad if batch smaller than BATCH_SIZE
      orig_batch_size = len(batch)
      if len(batch)<self.config.BATCH_SIZE:
        for x in range(self.config.BATCH_SIZE - len(batch)):
          batch = np.append(batch, batch[0])

      # Get input
      X, input_labels_social = get_inputs_outputs_vgg19full(np.array(self.dataset)[batch], self.explicit_edges)
      if ground_truth == 'human_ratings':
          input_labels = [mapping[x] for x in input_labels_social]
      else:
          input_labels = [mapping[x[0]] for x in input_labels_social] #only agent 0

      # feed dictionary
      feed_dict = {}
      feed_dict[self.X] = X
      feed_dict[self.target_V_placeholder] = input_labels

      # test
      test_values = self.sess.run({"loss": self.loss, "loss_V": self.loss_V,
                                   "output_label_V": self.output_label_V}, feed_dict)

      #print("Test Loss", test_values['loss'])
      test_loss += test_values['loss']
      test_loss_V += test_values['loss_V']

      correct_pred = np.equal(np.argmax(test_values['output_label_V'], 1), np.argmax(input_labels, 1))
      correct_pred = correct_pred[:orig_batch_size]
      accuracy_batch.append(np.mean(correct_pred))

      V0_pred.extend(np.argmax(test_values['output_label_V'][:orig_batch_size], 1))
      V0_true.extend(np.argmax(np.array(input_labels[:orig_batch_size]), 1))
      

    #print("Average Test Loss: ",test_loss/(len(V0_true)/self.config.BATCH_SIZE))
    #print("Average Test Loss V: ",test_loss_V/(len(V0_true)/self.config.BATCH_SIZE))
    print("Accuracy: ", np.mean(np.equal(V0_pred, V0_true)))
    print("Confusion Matrix Agent 0: \n",sklearn.metrics.confusion_matrix(V0_true, V0_pred))
    
    if output_predictions != False:
      return np.mean(np.equal(V0_pred, V0_true)), V0_true, V0_pred

    return np.mean(np.equal(V0_pred, V0_true))

  def cross_validate(self, n_splits, N_EPOCHS, X_train, y_train, mapping):
    skf = StratifiedKFold(n_splits)
    cross_val_acc = []
    for train, test in skf.split(X_train, y_train):   #when kf or rkf, then only X_train 
      self._initialize_session()
      self.train(N_EPOCHS,train_data_idx=np.array(X_train)[train], mapping=mapping, plot=False)
      cross_val_acc.append(self.test(test_data_idx=np.array(X_train)[test], mapping=mapping))
    return np.mean(cross_val_acc)

  def save_model(self, C_string):
    for i in range(len(self.config)):
      if isinstance(self.config[i],list):
        C_string =  C_string + "_" + '_'.join([str(x) for x in self.config[i][0]])
      else:
        C_string =  C_string + "_" + str(self.config[i])
    outfile = C_string + '/model'
    self.saver.save(self.sess, outfile)

  def load_model(self, C_string):
    #load from tf model
    for i in range(len(self.config)):
      if isinstance(self.config[i],list):
        C_string =  C_string + "_" + '_'.join([str(x) for x in self.config[i][0]])
      else:
        C_string =  C_string + "_" + str(self.config[i])
    infile = C_string + '/model.meta'
    load_saver = tf.train.import_meta_graph(infile)
    load_saver.restore(self.sess,tf.train.latest_checkpoint(C_string))

import pickle

model_config = namedtuple('model_config', 'FEATURE_SIZE V_OUTPUT_SIZE BATCH_SIZE CLASS_WEIGHTS LEARNING_RATE LAMBDA')

scores = {'cross_val':[], 'entire_trainset':[], 'testset':[]}

for split in range(bootstrap_splits):
  print("\n\n##### Bootstrap Split No.:", split)
  with open('./Gaze_Dataset/bootstrapped_traintest_splits_pickles/traintest_seqs_'+args.dataset + '_'+str(split)+'_vgg19full', 'rb') as f:
      Sequences = pickle.load(f)
      seq_train_idx = pickle.load(f)
      seq_test_idx = pickle.load(f)

  labels = [seq['label'][0] for seq in Sequences]

  if args.output_type == 2:
    mapping = {'AvertGaze':(1,0), 'GazeFollow': (1,0), 'JointAtt': (1,0), 'MutualGaze': (1,0), 'SingleGaze' : (0,1)}
    C = model_config(FEATURE_SIZE = 1500, V_OUTPUT_SIZE = 2, BATCH_SIZE = 20, CLASS_WEIGHTS = [[1.0,1.5]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
    map_2_social_nonsocial = {'AvertGaze':'Social', 'GazeFollow': 'Social', 'JointAtt': 'Social', 'MutualGaze': 'Social', 'SingleGaze' : 'NonSocial'}
    labels_for_skf = [map_2_social_nonsocial[x] for x in labels]  #binary labels
  elif args.output_type ==5:
    mapping = {'AvertGaze':(1,0,0,0,0), 'GazeFollow': (0,1,0,0,0), 'JointAtt': (0,0,1,0,0), 'MutualGaze': (0,0,0,1,0), 'SingleGaze' : (0,0,0,0,1)}
    C = model_config(FEATURE_SIZE = 1500, V_OUTPUT_SIZE = 5, BATCH_SIZE = 20, CLASS_WEIGHTS = [[5.69,4.42,1.85,1.66,1.0]], LEARNING_RATE = 1e-3, LAMBDA = 0.01 )
    labels_for_skf = labels

  N_EPOCHS = 150
  model = VGG19_wClassifier(Sequences, C, explicit_edges=False)
  model._initialize_session()

  if args.mode == "train":
    scores['cross_val'].append(model.cross_validate(5,N_EPOCHS, X_train=seq_train_idx, y_train=np.array(labels_for_skf)[seq_train_idx], mapping=mapping))
    scores['entire_trainset'].append(model.test(test_data_idx = seq_train_idx, mapping=mapping))
    scores['testset'].append(model.test(test_data_idx = seq_test_idx, mapping=mapping))
    print(scores)
    model.save_model(string + str(args.output_type) + '_' + str(split) + '_' + which_model)
  else:
    model.load_model(string + str(args.output_type) + '_' + str(split) + '_' + which_model)
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

