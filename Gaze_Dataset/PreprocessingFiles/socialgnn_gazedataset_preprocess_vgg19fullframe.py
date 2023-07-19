import os
import numpy as np
import pandas as pd
from collections import Counter
import cv2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

video_start_id = int(sys.argv[1])
no_videos_to_process = 100
path = '../annotation_cleaned'
n_videos = len(os.listdir(path))

D = []
D_dicts = {}

for root, dirs, files, in os.walk(path):
    for file in files:
      print(file)

      if file.endswith(".txt"):
        open_f = open(path + '/' + file, 'r')
        list_of_lists = []
        for line in open_f:
          stripped_line = line.strip()
          line_list = stripped_line.split()
          list_of_lists.append(line_list)

        d = pd.DataFrame(list_of_lists, columns=['bbx ID','xmin','ymin','xmax','ymax', 'frame ID', 'lost','occluded', 'generated','bbx label','event attribute','atomic attribute','attention focus'])
        print(Counter(d['bbx label']))

        d['frame ID'] = d['frame ID'].astype(int)
        d_dict = {k:g for k,g in sorted(d.groupby(by="frame ID"))}
        s = file.split('_')
        D_dicts[s[-1][:-4]] = d_dict


"""# **Frames to Features**"""


def frames_to_features_and_labels(annotations, video, pca, scaler):
  # load models
  base_model = VGG19(weights='imagenet')
  model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

  # Pass entities in videos through model to get features
  frames_features = []
  frames_labels = []
  frames_entities = []
  frames_edges = []

  curr_frame = len(frames_features)
  
  while True:
    ret, frame = video.read()

    if ret:
      frame = np.array(frame)
      
      temp = []
      temp_labels = []
      temp_entities = []
      temp_edges = []

      #print(curr_frame, len(frames_features))

      if curr_frame not in annotations.keys():
          curr_frame +=1
      else:
        x = np.expand_dims(frame, axis=0)
        x = preprocess_input(smart_resize(x, (224,224)))
        reshaped_frame = np.array(x)
        x_ = list(model.predict(x)[0])
        x_scaled = scaler.transform([x_])
        x_pca = pca.transform(x_scaled)
        x_pca = list(x_pca[0])


        for ind, entity in annotations[curr_frame].iterrows():
          temp_labels.append(entity['event attribute'])
          temp_entities.append(entity['bbx label'])
          temp_edges.append((entity['bbx label'], entity['attention focus']))

        frames_features.append(x_pca)
        frames_labels.append(temp_labels)
        frames_entities.append(temp_entities)
        frames_edges.append(temp_edges)
        curr_frame+=1

        print("frame labels", temp_labels)

      if len(frames_features) == len(annotations):
        break

  return frames_features, frames_labels, frames_entities, frames_edges



def extract_sequences(frames_features, frames_labels, frames_entities, frames_edges):
  sequences = []
  labels = []

  curr_seq = []
  for frame_id in range(len(frames_labels)):
    entities_and_labels = dict(zip(frames_entities[frame_id], frames_labels[frame_id]))

    if frame_id == 0:
      seq_entities_and_labels = entities_and_labels
    
    # if all entities in frame & social label matches
    if entities_and_labels == seq_entities_and_labels:
      # append frame to curr_seq
      curr_seq.append([frames_features[frame_id],frames_edges[frame_id]])
    else:
      # append curr_seq to sequences
      sequences.append(np.array(curr_seq))
      labels.append(seq_entities_and_labels)
      # create new seq and append frame to curr_seq
      seq_entities_and_labels = entities_and_labels
      curr_seq = []
      curr_seq.append([frames_features[frame_id],frames_edges[frame_id]])
    
  sequences.append(np.array(curr_seq))
  labels.append(seq_entities_and_labels)
  return sequences, labels

def get_all_edges(edge_combos, entities):
  senders = []
  receivers = []

  for a,b in edge_combos:
    if b!= None:
      if b[0] == 'O':
        b = b.replace('O', 'Object')
      if b[0] == 'P':
        b = b.replace('P', 'Person')

      if a in entities and b in entities and a!=b and a!="NA" and b!="NA":  
        senders.append(entities.index(a))
        receivers.append(entities.index(b))
  return senders,receivers

# No Entity type encoding: '0' for agent, '1' for object + Not truly a graph, just to keep it similar to code for SocialGNN and VisualRNN
def create_graphs_1video_1t(features, entities, edges):
    nodes = features

    #calculate edge or not; undirected edges so add to both senders & receivers
    senders, receivers = get_all_edges(edges, entities)

    #create graph
    graph_dict = {"nodes": nodes, "senders": senders, "receivers": receivers}  #No edge features or global features
    return graph_dict

import itertools
import pickle
out = D_dicts #dict(itertools.islice(D_dicts.items(), 2)) 

with open('../preprocessed_pickles/fitted_PCA_vgg19full', "rb") as f:
  pca = pickle.load(f)
  scaler = pickle.load(f)


# Get features from videos
V = {}
for i in range(video_start_id, video_start_id + no_videos_to_process):
    if i >= len(D_dicts):
       break
    k = list(D_dicts)[i]
    print("Video:",k)
    video = cv2.VideoCapture('../video/' + str(k)+ '.mp4')
    annotations = D_dicts[k]

    frames_features, frames_labels, frames_entities, frames_edges = frames_to_features_and_labels(annotations, video, pca, scaler)
    print(len(frames_features))
    print("check", len(frames_features), len(frames_features[0]), frames_labels[0])
    print(len(frames_edges), frames_edges[0])

    # extract sequences
    sequences, labels = extract_sequences(frames_features, frames_labels, frames_entities, frames_edges)
    print(len(sequences), labels)

    # create graph
    sequence_no = 0
    frame_no = 0

    graphs_allsequences = []
    for sequence_no in range(len(sequences)):
      graphs_1sequence = []
      for frame_no in range(len(sequences[sequence_no])):
        graphs_1sequence.append(create_graphs_1video_1t(sequences[sequence_no][frame_no][0], list(labels[sequence_no].keys()), sequences[sequence_no][frame_no][1]))
      graphs_allsequences.append(graphs_1sequence)

    V[k] = {"graph_dicts": graphs_allsequences, "sequences": sequences, "labels": labels}

f_name = '../preprocessed_pickles/processed_vgg19full_Jun5_'+ str(video_start_id)
with open(f_name, 'wb') as file:
    pickle.dump(V, file)


