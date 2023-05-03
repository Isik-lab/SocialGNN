### PREREQUISITES

import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
from itertools import combinations
import itertools 
import matplotlib.pyplot as plt
from collections import Counter
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold

ground_truth = "human_ratings"

### HELPER FUNCTIONS

# Input to specify what enity IDs are agents
agents = [0,1]

# Graph Creationg Function Definitions
def get_entity_colors(entity_color_code):
  entity_color_map = {}
  v = list(entity_color_code.values())
  if v[0] == 0:
    entity_color_map['agent0'] = 'red'
    entity_color_map['agent1'] = 'green'
  else:
    entity_color_map['agent0'] = 'green'
    entity_color_map['agent1'] = 'red'
  
  if v[-1] == 3:
    entity_color_map['item0'] = 'lightblue'
    entity_color_map['item1'] = 'pink'
  else:
    entity_color_map['item0'] = 'pink'
    entity_color_map['item1'] = 'lightblue'
  return entity_color_map

# A Python3 program to find if 2 given line segments intersect or not: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/ (modified)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
  
# Given three colinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False
  
def orientation(p, q, r):
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
      return 1 #clockwise
    elif (val < 0):
      return 2 #counterclockwise
    else:
      return 0 #collinear
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(wall_start,wall_end,obj1_pos,obj2_pos): 
    p1 = Point(wall_start[0], wall_start[1])
    q1 = Point(wall_end[0], wall_end[1])
    p2 = Point(obj1_pos[0], obj1_pos[1])
    q2 = Point(obj2_pos[0], obj2_pos[1])
    # Find the 4 orientations required for the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    ### General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    #### Special Cases
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
    # If none of the cases
    return False

def contact_based_edge(obj1_pos, obj2_pos, obj1_size, obj2_size, wall_segs):
  d = np.sqrt((np.sum(obj1_pos[0]-obj2_pos[0])**2+(np.sum(obj1_pos[1]-obj2_pos[1])**2)))
  contact = False
  if d < obj1_size + obj2_size:
    contact = True

  wall_between = False
  for wall_no in range(len(wall_segs)):
    if doIntersect(wall_segs[wall_no][0],wall_segs[wall_no][1], obj1_pos, obj2_pos) == True:
      wall_between = True
  
  if contact == True and wall_between == False:
      return 1
  return 0

# Return senders and receivers for a graph
def get_all_edges(nodes, trajectories, sizes, wall_segs, t):
  senders = []
  receivers = []
  
  for entity1,entity2 in combinations(nodes,2):
    obj1_pos = trajectories[entity1][t][:2]
    obj2_pos = trajectories[entity2][t][:2]
    obj1_size = sizes[entity1]*1.2    #factor of 2?
    obj2_size = sizes[entity2]*1.2
    e = contact_based_edge(obj1_pos, obj2_pos, obj1_size, obj2_size, wall_segs)
    if e == 1:
      senders.append(entity1)
      receivers.append(entity2)
      senders.append(entity2)
      receivers.append(entity1)
  return senders,receivers

# Create graphs for all timesteps of a video
# Entity type encoding: '0' for agent, '1' for object
def create_graphs_1video_1t(n_entities,t, trajectories, entity_sizes, wall_segs, landmark_centers):
    nodes = []
    # Node features : pos_x, pos_y, vel_x, vel_y, angle, size, entity_type
    for entity in range(n_entities):
      features = list(trajectories[entity][t])
      features.append(entity_sizes[entity])   #although size doesn't change as a function of time
      if entity in agents:    #entity type encoding
        features.append(0)    
      else:
        features.append(1)
      nodes.append(features)

    #calculate edge or not; undirected edges so add to both senders & receivers
    senders, receivers = get_all_edges(range(n_entities),trajectories,entity_sizes,wall_segs,t)

    #global variable
    wall_segs_pos = [[(0.,0.),(0.,0.)],[(0.,0.),(0.,0.)],[(0.,0.),(0.,0.)],[(0.,0.),(0.,0.)]]
    for j in range(len(wall_segs)):
      wall_segs_pos[j]=wall_segs[j]
    u = np.float32(np.array(wall_segs_pos).flatten())
    u = np.float32(np.append(u, landmark_centers).flatten())

    #create graph
    graph_dict = {"nodes": nodes, "senders": senders, "receivers": receivers, "globals": u}  #No edge features 
    return graph_dict

# Create graph dicts from a set of Input videos (graphs for all timesteps of all videos)
def create_Gin(InpVideos):
  graph_dicts_list = []
  video_timesteps = []
  for v in range(len(InpVideos)):
    n_entities = len(InpVideos[v]['trajectories']) #-1 if remove last e from trajectories when averaging steps   
    n_timesteps = len(InpVideos[v]['trajectories'][0])  
    landmark_centers = InpVideos[v]['landmark_centers']
    for t in range(n_timesteps):   
      graph_dict = create_graphs_1video_1t(n_entities, t, InpVideos[v]['trajectories'], InpVideos[v]['entity_sizes'], InpVideos[v]['wall_segs'], InpVideos[v]['landmark_centers'])
      graph_dicts_list.append(graph_dict)
    video_timesteps.append(n_timesteps)
  return graph_dicts_list, video_timesteps

### MODEL INPUT OUTPUT CREATION FUNCTIONS

def get_inputs_outputs(InpVideos):
  graph_dicts_list, videos_timesteps = create_Gin(InpVideos)
  labels_social = []
  for v in range(len(InpVideos)):
    if ground_truth == "human_ratings":
      labels_social.append(InpVideos[v]['social_goals']) 
    else:
      labels_social.append([InpVideos[v]['social_goals'][a][0] for a in agents])
  return graph_dicts_list, videos_timesteps, labels_social

def get_edges_boolean(Gin, E_SPATIAL_SIZE):
  s_ = [0,0,0,1,1,1,2,2,2,3,3,3]
  r_ = [1,2,3,0,2,3,0,1,3,0,1,2]

  edges_boolean_dict = []

  for graph_id in range(len(Gin.n_edge)):
    d = {(sx,rx):False for sx,rx in zip(s_,r_)}
    for j in range(Gin.n_edge[graph_id]):
      ind = j+sum(Gin.n_edge[:graph_id])
      sx = Gin.senders[ind]-4*graph_id    #mapping to actual node ID
      rx = Gin.receivers[ind]-4*graph_id  #mapping to actual node ID
      d[(sx,rx)]=True
    edges_boolean_dict.append(list(d.values()))

  edges_boolean = list(np.concatenate(edges_boolean_dict).flat)
  edges_boolean = list(itertools.chain.from_iterable(itertools.repeat(x,  E_SPATIAL_SIZE) for x in edges_boolean))
  return edges_boolean

### SocialGNN V_Pred MODEL

class SocialGNN(object):
  def __init__(self, dataset, config, context_info, sample_graph_dicts_list):
    self.graph = tf.Graph()

    self.dataset = dataset
    self.config = config  #define model parameters
    self.context_info = context_info

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
    Gspatial_nodes = gn.blocks.NodeBlock(node_model_fn=lambda: snt.nets.MLP([self.config.V_SPATIAL_SIZE]), use_globals=self.context_info)   #REVISIT #n_nodes(4)x timesteps(101/61) x n_videos(20)
    Gtemporal_nodes = snt.LSTM(hidden_size=self.config.V_TEMPORAL_SIZE)
    classifier_nodes = snt.Linear(self.config.V_OUTPUT_SIZE)

    #########   Create graph    #########

    #########   Spatial: Nodes & Edges    #########
    G_E = Gspatial_edges(self.Gin_placeholder)
    G_V = Gspatial_nodes(G_E)

    #########   Temporal Nodes    #########
    x_reshaped = tf.reshape(G_V.nodes, [-1,self.config.NUM_NODES,self.config.V_SPATIAL_SIZE])
    x_reshaped_sliced = x_reshaped[:,:self.config.NUM_AGENTS,:]   #only agents
    x_reshaped_sliced_reshaped = tf.reshape(x_reshaped_sliced, [-1, self.config.NUM_AGENTS*self.config.V_SPATIAL_SIZE]) #concat features
    V_tensor = tf.RaggedTensor.from_row_lengths(x_reshaped_sliced_reshaped,row_lengths=self.videos_timesteps_placeholder ) #separate videowise timesteps
    V_tensor = V_tensor.to_tensor()

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
    self.final_state = final_state[0] #final (non-zero) layer of dynamicRNN

  def train(self, N_EPOCHS, train_data_idx, mapping, plot=True):
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

  def get_activations(self, test_data_idx, mapping, type = "RNN"):
    print("\n.............TESTING..............")
    test_loss = 0
    test_loss_V = 0
    RNN_activations = []

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
          "output_label_V": self.output_label_V, "RNN_activations": self.final_state}, feed_dict)

      #print("Test Loss", test_values['loss'])
      test_loss += test_values['loss']

      if type == "RNN":
        RNN_activations.extend(test_values["RNN_activations"])
      elif type == "classifier":
        RNN_activations.extend(test_values["output_label_V"])

    return RNN_activations

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


### SocialGNN E_Pred Model
class SocialGNN_E(object):
  def __init__(self, dataset, config, context_info, sample_graph_dicts_list, ablate = False):
    self.graph = tf.Graph()

    self.dataset = dataset
    self.config = config  #define model parameters
    self.context_info = context_info
    self.ablate = ablate

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
    self.target_E_placeholder = tf.placeholder(tf.float32,shape=[None, self.config.E_OUTPUT_SIZE], name='target_E')
    self.videos_timesteps_placeholder = tf.placeholder(tf.int32,shape=[None], name='videos_timesteps')
    self.edges_boolean = tf.placeholder(tf.bool,shape=[None], name='edges_boolean')

  def _build_graph(self):
    print("\n.............BUILDING GRAPH..............")
    #########   Define Layers/Blocks    #########
    Gspatial_edges = gn.blocks.EdgeBlock(edge_model_fn=lambda: snt.Linear(self.config.E_SPATIAL_SIZE), use_globals=self.context_info, use_edges = False)   #no edge attributes used #n_edges(unequal) x timesteps(101/61) x n_videos(20)
    Gtemporal_edges = snt.LSTM(hidden_size=self.config.E_TEMPORAL_SIZE)  #https://github.com/deepmind/graph_nets/issues/52 GN doesnt support recurrent locks yet..
    classifier_edges = snt.Linear(self.config.E_OUTPUT_SIZE)

    #########   Create graph    #########

    #########   Spatial: Nodes & Edges    #########
    G_E = Gspatial_edges(self.Gin_placeholder)

    #########   Temporal Edges    #########
    idx_keep = tf.where(self.edges_boolean)[:,-1]
    idx_remove = tf.where(tf.equal(self.edges_boolean,False))[:,-1]
    values_remove = tf.zeros((tf.shape(idx_remove)[0]))
    values_keep = tf.cast(tf.reshape(G_E.edges,[-1]), tf.float32)
    zeros_remove = tf.zeros_like(idx_remove) # to create a sparse vector we still need 2d indices like [ [0,1], [0,2], [0,10] ]
    zeros_keep = tf.zeros_like(idx_keep) # create vectors of 0's that we'll later stack with the actual indices
    idx_remove = tf.stack( [ zeros_remove, idx_remove], axis=1 )
    idx_keep = tf.stack( [ zeros_keep, idx_keep], axis=1 )
    logits_remove = tf.SparseTensor( idx_remove, values_remove, dense_shape=[1, tf.shape(self.edges_boolean)[0]])
    logits_keep = tf.SparseTensor( idx_keep, values_keep, dense_shape=[1,tf.shape(self.edges_boolean)[0]])
    filtered_logits = tf.add(tf.sparse.to_dense(logits_remove, default_value = 0. ),tf.sparse.to_dense(logits_keep, default_value = 0. ))

    x_padded = tf.reshape(filtered_logits, [-1,self.config.MAX_EDGES, self.config.E_SPATIAL_SIZE] ) #n_graphs x 12 possible edges x 5 dim edgefeatures
    #print(x_padded.shape)
    if self.ablate:
      x_padded_sliced = tf.reshape(tf.gather(x_padded,indices=[0,3], axis=1),[-1,2*self.config.E_SPATIAL_SIZE] )
    else:
      x_padded_sliced = tf.reshape(tf.gather(x_padded,indices=[0,1,2,3,4,5,6,7,8,9,10,11], axis=1),[-1,12*self.config.E_SPATIAL_SIZE] )
    #print(x_padded_sliced.shape)
    x_final = tf.RaggedTensor.from_row_lengths(x_padded_sliced,row_lengths=self.videos_timesteps_placeholder) #separate videowise timesteps

    E_tensor = x_final.to_tensor()
    E_tensor = tf.cast(E_tensor, tf.float32)
    #print("E tensor",E_tensor.shape)
    

    # RNN
    output_sequence, final_state = tf.nn.dynamic_rnn(Gtemporal_edges, E_tensor, self.videos_timesteps_placeholder, Gtemporal_edges.zero_state(self.config.BATCH_SIZE, tf.float32))
    #print("LSTM out", output_sequence)
    # Classify
    output_label_E = classifier_edges(final_state[0]) 
    #print("Output E",output_label_E)

    #########   Training Loss + Optimizer    #########
    weights = tf.reduce_sum(self.config.CLASS_WEIGHTS * self.target_E_placeholder, axis=1)  # deduce weights for batch samples based on their true label
    self.loss_E = tf.losses.softmax_cross_entropy(self.target_E_placeholder, output_label_E, weights = weights)
    self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.config.LAMBDA
    self.loss = self.loss_E + self.lossL2 
    self.optimizer = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
    self.step_op = self.optimizer.minimize(self.loss)
    #print("\nLoss", self.loss_E, self.loss)
    #print("Optimizer",self.optimizer)

    self.output_label_E = output_label_E
    #print("\nTrainable paramters: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    self.trainable_variables = tf.trainable_variables()
    self.final_state = final_state[0] #final (non-zero) layer of dynamicRNN

  def train(self, N_EPOCHS, train_data_idx, mapping, plot=False):
    print("\n.............TRAINING..............")
    training_loss = {'total_loss':[], 'loss_E': []}
    for e in range(N_EPOCHS):
      np.random.shuffle(train_data_idx)
      batches = [train_data_idx[k:k+self.config.BATCH_SIZE] for k in range(0, len(train_data_idx), self.config.BATCH_SIZE)]

      epoch_loss = 0
      epoch_loss_E = 0
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
        feed_dict[self.edges_boolean] = get_edges_boolean(Gin, self.config.E_SPATIAL_SIZE)
        feed_dict[self.target_E_placeholder] = input_labels

        # train
        train_values = self.sess.run({"step": self.step_op,
          "loss": self.loss, "loss_E": self.loss_E,
           "output_label_E": self.output_label_E}, feed_dict)
        
        epoch_loss += train_values['loss']
        epoch_loss_E += train_values['loss_E']

      print("Epoch No.:",e,"\tLoss:",epoch_loss/len(batches), "\tLoss_E:",epoch_loss_E/len(batches))
      training_loss['total_loss'].append(epoch_loss/len(batches))
      training_loss['loss_E'].append(epoch_loss_E/len(batches))

    if plot==True:
      plt.figure(figsize=(10,7))
      plt.plot(training_loss['total_loss'], label='total loss')
      plt.plot(training_loss['loss_E'], label = 'loss_E')
      plt.legend()
      plt.ylabel('Training Loss')
      plt.xlabel('Number of Epochs')
      plt.show()

  def test(self, test_data_idx, mapping, output_predictions = False):
    print("\n.............TESTING..............")
    test_loss = 0
    test_loss_E = 0
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
      feed_dict[self.edges_boolean] = get_edges_boolean(Gin, self.config.E_SPATIAL_SIZE)
      feed_dict[self.target_E_placeholder] = input_labels

      # test
      test_values = self.sess.run({"loss": self.loss, "loss_E":self.loss_E,
          "output_label_E": self.output_label_E}, feed_dict)

      #print("Test Loss", test_values['loss'])
      test_loss += test_values['loss']

      correct_pred = np.equal(np.argmax(test_values['output_label_E'], 1), np.argmax(input_labels, 1))
      correct_pred = correct_pred[:orig_batch_size]
      accuracy_batch.append(np.mean(correct_pred))

      V0_pred.extend(np.argmax(test_values['output_label_E'][:orig_batch_size], 1))
      V0_true.extend(np.argmax(np.array(input_labels[:orig_batch_size]), 1))
      
    #print("Average Test Loss: ",test_loss/(len(V0_true)/self.config.BATCH_SIZE))
    print("Accuracy: ", np.mean(np.equal(V0_pred, V0_true)))
    print("Confusion Matrix Agent 0: \n",sklearn.metrics.confusion_matrix(V0_true, V0_pred))
    
    if output_predictions != False:
      return np.mean(np.equal(V0_pred, V0_true)), V0_true, V0_pred

    return np.mean(np.equal(V0_pred, V0_true))

  def get_activations(self, test_data_idx, mapping, type = "RNN"):
    print("\n.............TESTING..............")
    test_loss = 0
    test_loss_V = 0
    RNN_activations = []

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
      feed_dict[self.edges_boolean] = get_edges_boolean(Gin, self.config.E_SPATIAL_SIZE)
      feed_dict[self.target_E_placeholder] = input_labels

      # test
      test_values = self.sess.run({"loss": self.loss, 
          "output_label_E": self.output_label_E, "RNN_activations": self.final_state}, feed_dict)

      #print("Test Loss", test_values['loss'])
      test_loss += test_values['loss']

      if type == "RNN":
        RNN_activations.extend(test_values["RNN_activations"])
      elif type == "classifier":
        temp = [np.exp(x)/sum(np.exp(x)) for x in test_values["output_label_E"]]
        RNN_activations.extend(temp)

    return RNN_activations

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
  def __init__(self, dataset, config, context_info = False, explicit_edges=False):
    self.graph = tf.Graph()

    self.dataset = dataset
    self.config = config  #define model parameters
    self.explicit_edges = explicit_edges
    self.context_info = context_info

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
    self.final_state = final_state[0] #final (non-zero) layer of dynamicRNN
    #print("\nLoss", self.loss_V, self.loss)
    #print("Optimizer",self.optimizer)


  def train(self, N_EPOCHS, train_data_idx, mapping, plot=True):
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
        X, input_videos_timesteps, input_labels_social = get_inputs_outputs_baseline(np.array(self.dataset)[batch], self.explicit_edges, self.context_info)
        if ground_truth == 'human_ratings':
          input_labels = [mapping[x] for x in input_labels_social]
        else:
          input_labels = [mapping[x[0]] for x in input_labels_social] #only agent 0

        # feed dictionary
        feed_dict = {}
        feed_dict[self.X] = X
        feed_dict[self.videos_timesteps_placeholder] = input_videos_timesteps
        feed_dict[self.target_V_placeholder] = input_labels

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
      X, input_videos_timesteps, input_labels_social = get_inputs_outputs_baseline(np.array(self.dataset)[batch], self.explicit_edges, self.context_info)
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

  def get_activations(self, test_data_idx, mapping):
    print("\n.............TESTING..............")
    test_loss = 0
    test_loss_V = 0
    RNN_activations = []

    batches = [test_data_idx[k:k+self.config.BATCH_SIZE] for k in range(0, len(test_data_idx), self.config.BATCH_SIZE)]
    for batch in batches:
      # pad if batch smaller than BATCH_SIZE
      orig_batch_size = len(batch)
      if len(batch)<self.config.BATCH_SIZE:
        for x in range(self.config.BATCH_SIZE - len(batch)):
          batch = np.append(batch, batch[0])

      # Get input
      X, input_videos_timesteps, input_labels_social = get_inputs_outputs_baseline(np.array(self.dataset)[batch], self.explicit_edges, self.context_info)
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
      test_values = self.sess.run({"loss": self.loss, "loss_V":self.loss_V,
          "output_label_V": self.output_label_V, "RNN_activations": self.final_state}, feed_dict)

      #print("Test Loss", test_values['loss'])
      test_loss += test_values['loss']

      RNN_activations.extend(test_values["RNN_activations"])

    return RNN_activations

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


def get_inputs_outputs_baseline(InpVideos, explicit_edges=False, context_info = True):
  X = []
  video_timesteps = []
  labels_social = []
  for v in range(len(InpVideos)):
    n_entities = len(InpVideos[v]['trajectories']) #-1 if remove last e from trajectories when averaging steps   
    n_timesteps = len(InpVideos[v]['trajectories'][0])  

    video_timesteps.append(n_timesteps)
    
    if ground_truth == "human_ratings":
      labels_social.append(InpVideos[v]['social_goals']) 
    else:
      labels_social.append([InpVideos[v]['social_goals'][a][0] for a in agents]) 
    
    for t in range(n_timesteps):  
      nodes = []
      # Node features : pos_x, pos_y, vel_x, vel_y, angle, size, entity_type
      for entity in range(n_entities):
        features = list(InpVideos[v]['trajectories'][entity][t])
        features.append(InpVideos[v]['entity_sizes'][entity])   #although size doesn't change as a function of time
        if entity in agents:    #entity type encoding
          features.append(0)    
        else:
          features.append(1)
        nodes.extend(features)    #extended features

      
      #explicit edges
      if explicit_edges == True:
        senders, receivers = get_all_edges(range(n_entities),InpVideos[v]['trajectories'],InpVideos[v]['entity_sizes'],InpVideos[v]['wall_segs'],t)
        s_ = [0,0,0,1,1,1,2,2,2,3,3,3]
        r_ = [1,2,3,0,2,3,0,1,3,0,1,2]
        d = {(sx,rx):0 for sx,rx in zip(s_,r_)}
        for edge in range(len(senders)):
          d[(senders[edge],receivers[edge])] = 1
        edges = np.float32(list(d.values()))
        nodes.extend(edges)
      
      
      #global variable
      if context_info == True:
        wall_segs = InpVideos[v]['wall_segs']
        wall_segs_pos = [[(0.,0.),(0.,0.)],[(0.,0.),(0.,0.)],[(0.,0.),(0.,0.)],[(0.,0.),(0.,0.)]]
        for j in range(len(wall_segs)):
          wall_segs_pos[j]=wall_segs[j]
        u = np.float32(np.array(wall_segs_pos).flatten())
        u = np.float32(np.append(u, InpVideos[v]['landmark_centers']).flatten())
        nodes.extend(u)  
      

      X.append(nodes)
  return X, video_timesteps, labels_social
