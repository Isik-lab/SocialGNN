import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import sklearn
from collections import Counter

#changed get_input_outputs() function and arguments
ground_truth = "given_labels"

def get_inputs_outputs(Sequences):
  graph_dicts_list = []
  video_timesteps = []
  labels_social = []
  for seq in Sequences:
    graph_dicts_list.extend(seq['graph_dicts'])
    video_timesteps.append(len(seq['graph_dicts']))
    labels_social.append(seq['label'])
  return graph_dicts_list, video_timesteps, labels_social


class SocialGNN(object):
  def __init__(self, dataset, config, sample_graph_dicts_list):
    self.graph = tf.Graph()

    self.dataset = dataset
    self.config = config  #define model parameters

    with self.graph.as_default():
      self._define_inputs(sample_graph_dicts_list)
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

  def _define_inputs(self, sample_graph_dicts_list):
    print(".............DEFINING INPUT PLACEHOLDERS..............")
    #########   Create input placeholder    #########
    self.Gin_placeholder = gn.utils_tf.placeholders_from_data_dicts(sample_graph_dicts_list)
    self.target_V_placeholder = tf.placeholder(tf.float32,shape=[None, self.config.V_OUTPUT_SIZE], name='target_V') 
    self.videos_timesteps_placeholder = tf.placeholder(tf.int32,shape=[None], name='videos_timesteps')


  def _build_graph(self):
    print("\n.............BUILDING GRAPH..............")
    #########   Define Layers/Blocks    #########
    Gspatial_edges = gn.blocks.EdgeBlock(edge_model_fn=lambda: snt.Linear(self.config.E_SPATIAL_SIZE), use_globals=False, use_edges = False)   #no edge attributes used #n_edges(unequal) x timesteps(101/61) x n_videos(20)
    Gspatial_nodes = gn.blocks.NodeBlock(node_model_fn=lambda: snt.Linear(self.config.V_SPATIAL_SIZE), use_globals=False)   #REVISIT #n_nodes(4)x timesteps(101/61) x n_videos(20)
    Gtemporal_nodes = snt.LSTM(hidden_size=self.config.V_TEMPORAL_SIZE)
    classifier_nodes = snt.Linear(self.config.V_OUTPUT_SIZE)

    #########   Create graph    #########

    #########   Spatial: Nodes & Edges    #########
    G_E = Gspatial_edges(self.Gin_placeholder)
    G_V = Gspatial_nodes(G_E)
    #print("\nG_E",G_E,"\nG_V",G_V)

    #########   Temporal Nodes    #########
    x_reshaped = tf.reshape(G_V.nodes, [-1,self.config.NUM_NODES,self.config.V_SPATIAL_SIZE])
    x_reshaped_sliced = x_reshaped[:,:self.config.NUM_AGENTS,:]   #only agents
    x_reshaped_sliced_reshaped = tf.reshape(x_reshaped_sliced, [-1, self.config.NUM_AGENTS*self.config.V_SPATIAL_SIZE]) #concat features
    V_tensor = tf.RaggedTensor.from_row_lengths(x_reshaped_sliced_reshaped,row_lengths=self.videos_timesteps_placeholder ) #separate videowise timesteps
    V_tensor = V_tensor.to_tensor()
    #print("\nV_tensor",V_tensor)

    # RNN
    output_sequence, final_state = tf.nn.dynamic_rnn(Gtemporal_nodes, V_tensor, self.videos_timesteps_placeholder, Gtemporal_nodes.zero_state(self.config.BATCH_SIZE, tf.float32))
    # Classify
    output_label_V = classifier_nodes(final_state[0])  
    #print("LSTM out", output_sequence,"\nOutput V",output_label_V)

    #########   Training Loss + Optimizer    #########
    weights = tf.reduce_sum(self.config.CLASS_WEIGHTS * self.target_V_placeholder, axis=1)  # deduce weights for batch samples based on their true label
    self.loss_V = tf.losses.softmax_cross_entropy(self.target_V_placeholder, output_label_V, weights = weights)
    self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.config.LAMBDA
    self.loss = self.loss_V + self.lossL2 
    self.optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
    self.step_op = self.optimizer.minimize(self.loss)
    #print("\nLoss", self.loss_V, self.loss)
    #print("Optimizer",self.optimizer)

    self.output_label_V = output_label_V
    #print("\nTrainable paramters: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    self.trainable_variables = tf.trainable_variables()

  def train(self, N_EPOCHS, train_data_idx, mapping, plot=False):
    print("\n.............TRAINING..............")
    training_loss = {'total_loss':[], 'loss_V': []}
    for e in range(N_EPOCHS):
      np.random.shuffle(train_data_idx)
      batches = [train_data_idx[k:k+self.config.BATCH_SIZE] for k in range(0, len(train_data_idx), self.config.BATCH_SIZE)]

      epoch_loss = 0
      epoch_loss_V = 0
      for batch in batches:
        # pad if batch smaller than BATCH_SIZE
        if len(batch)<self.config.BATCH_SIZE:
          for x in range(self.config.BATCH_SIZE - len(batch)):
            batch = np.append(batch, batch[0])
        
        # get X_train and Y_train from dataset + train_idx
        input_graph_dicts_list, input_videos_timesteps, input_labels_social = get_inputs_outputs(np.array(self.dataset)[batch])
        Gin = gn.utils_np.data_dicts_to_graphs_tuple(input_graph_dicts_list)
        if ground_truth == 'human_ratings':
          input_labels = [mapping[x] for x in input_labels_social]
        else:
          input_labels = [mapping[x[0]] for x in input_labels_social] #only agent 0

        # feed dictionary
        feed_dict = gn.utils_tf.get_feed_dict(self.Gin_placeholder, Gin)  #needed because None fields
        feed_dict[self.videos_timesteps_placeholder] = input_videos_timesteps
        feed_dict[self.target_V_placeholder] = input_labels
        #feed_dict[self.edges_boolean] = get_edges_boolean(Gin, self.config.E_SPATIAL_SIZE)

        # train
        train_values = self.sess.run({"step": self.step_op,
          "loss": self.loss, "loss_V": self.loss_V,
           "output_label_V": self.output_label_V}, feed_dict)
        
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

  def test(self, test_data_idx, mapping, output_predictions = False):
    print("\n.............TESTING..............")
    test_loss = 0
    test_loss_V = 0
    accuracy_batch = []
    V0_pred = []
    V0_true = []
    V1_pred = []
    V1_true = []

    batches = [test_data_idx[k:k+self.config.BATCH_SIZE] for k in range(0, len(test_data_idx), self.config.BATCH_SIZE)]
    for batch in batches:
      # pad if batch smaller than BATCH_SIZE
      orig_batch_size = len(batch)
      if len(batch)<self.config.BATCH_SIZE:
        for x in range(self.config.BATCH_SIZE - len(batch)):
          batch = np.append(batch, batch[0])
      
      # Get input
      input_graph_dicts_list, input_videos_timesteps, input_labels_social = get_inputs_outputs(np.array(self.dataset)[batch])
      Gin = gn.utils_np.data_dicts_to_graphs_tuple(input_graph_dicts_list)
      if ground_truth == 'human_ratings':
          input_labels = [mapping[x] for x in input_labels_social]
      else:
          input_labels = [mapping[x[0]] for x in input_labels_social] #only agent 0

      # feed dictionary
      feed_dict = gn.utils_tf.get_feed_dict(self.Gin_placeholder, Gin)  #needed because None fields
      feed_dict[self.videos_timesteps_placeholder] = input_videos_timesteps
      feed_dict[self.target_V_placeholder] = input_labels
      #feed_dict[self.edges_boolean] = get_edges_boolean(Gin, self.config.E_SPATIAL_SIZE)

      # test
      test_values = self.sess.run({"loss": self.loss, "loss_V":self.loss_V,
          "output_label_V": self.output_label_V}, feed_dict)

      #print("Test Loss", test_values['loss'])
      test_loss += test_values['loss']

      correct_pred = np.equal(np.argmax(test_values['output_label_V'], 1), np.argmax(input_labels, 1))
      correct_pred = correct_pred[:orig_batch_size]
      accuracy_batch.append(np.mean(correct_pred))

      V0_pred.extend(np.argmax(test_values['output_label_V'][:orig_batch_size], 1))
      V0_true.extend(np.argmax(np.array(input_labels[:orig_batch_size]), 1))

    #print("Average Test Loss: ",test_loss/(len(V0_true)/self.config.BATCH_SIZE))
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


### Baseline: Cue-based LSTM Model

class CueBasedLSTM(object):
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
    self.videos_timesteps_placeholder = tf.placeholder(tf.int32,shape=[None])
    self.target_V_placeholder = tf.placeholder(tf.float32,shape=[None, self.config.V_OUTPUT_SIZE])

  def _build_graph(self):
    print("\n.............BUILDING GRAPH..............")
    #########   Define Layers/Blocks    #########
    LSTM = snt.LSTM(hidden_size=self.config.V_TEMPORAL_SIZE)
    classifier = snt.Linear(self.config.V_OUTPUT_SIZE)

    #########   Create graph    #########
    X_ragged = tf.RaggedTensor.from_row_lengths(self.X, row_lengths=self.videos_timesteps_placeholder).to_tensor()
    # should i fix shape here?
    output_sequence, final_state = tf.nn.dynamic_rnn(LSTM, X_ragged, self.videos_timesteps_placeholder, LSTM.zero_state(self.config.BATCH_SIZE, tf.float32))
    output_label_V = classifier(final_state[0])  
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
        X, input_videos_timesteps, input_labels_social = get_inputs_outputs_baseline(np.array(self.dataset)[batch], self.explicit_edges)
        if ground_truth == 'human_ratings':
          input_labels = [mapping[x] for x in input_labels_social]
        else:
          input_labels = [mapping[x[0]] for x in input_labels_social] #only agent 0

        # feed dictionary
        feed_dict = {}
        feed_dict[self.X] = X
        feed_dict[self.videos_timesteps_placeholder] = input_videos_timesteps
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
      X, input_videos_timesteps, input_labels_social = get_inputs_outputs_baseline(np.array(self.dataset)[batch], self.explicit_edges)
      if ground_truth == 'human_ratings':
          input_labels = [mapping[x] for x in input_labels_social]
      else:
          input_labels = [mapping[x[0]] for x in input_labels_social] #only agent 0

      # feed dictionary
      feed_dict = {}
      feed_dict[self.X] = X
      feed_dict[self.videos_timesteps_placeholder] = input_videos_timesteps
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

import itertools
def get_inputs_outputs_baseline(Sequences, explicit_edges = False):
  seq_features_list = []
  labels_social = []
  video_timesteps = []

  for seq in Sequences:
    #curr_seq_features = [np.concatenate(frame['nodes']).tolist() for frame in seq['graph_dicts']]
    curr_seq_features = []
    for frame in seq['graph_dicts']:
      nodes = np.concatenate(frame['nodes']).tolist()

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

    seq_features_list.extend(curr_seq_features)
    labels_social.append(seq['label'])
    video_timesteps.append(len(seq['graph_dicts']))
  return seq_features_list, video_timesteps, labels_social