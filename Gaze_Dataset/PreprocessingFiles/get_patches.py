import os
import numpy as np
import pandas as pd

def load_annotations(path):
  D_dicts = {}
  for root, dirs, files, in os.walk(path):
      for file in files:
        #print(file)

        if file.endswith(".txt"):
          open_f = open(path + '/' + file, 'r')
          list_of_lists = []
          for line in open_f:
            stripped_line = line.strip()
            line_list = stripped_line.split()
            list_of_lists.append(line_list)

          #print(len(list_of_lists),list_of_lists)
          d = pd.DataFrame(list_of_lists, columns=['bbx ID','xmin','ymin','xmax','ymax', 'frame ID', 'lost','occluded', 'generated','bbx label','event attribute','atomic attribute','attention focus'])
          #print(Counter(d['bbx label']))

          d['frame ID'] = d['frame ID'].astype(int)
          d_dict = {k:g for k,g in sorted(d.groupby(by="frame ID"))}
          s = file.split('_')
          D_dicts[s[-1][:-4]] = d_dict
          #print(s[-1][:-4])
  return D_dicts


path = '../annotation_cleaned'

D_dicts = load_annotations(path)


import cv2

def get_video_patches(dataset):
  all_patches = []
  for k,v in dataset.items():
    print("Video:",k)
    video = cv2.VideoCapture('../video/' + str(k)+ '.mp4')
    annotations = v

    print("annotation keys", list(annotations.keys())[-1], len(annotations.keys()))

    curr_frame = 0
   
    while True:
      ret, frame = video.read()
  
      if ret:
        frame = np.array(frame)
        
        if curr_frame not in annotations.keys():
          curr_frame +=1
        else:
          for ind, entity in annotations[curr_frame].iterrows():
            x = frame[int(entity['ymin']):int(entity['ymax']),int(entity['xmin']):int(entity['xmax'])]
            all_patches.append(np.array(x))
          curr_frame+=1
      else:
        break

  return all_patches

import pickle

#import itertools
out = D_dicts #dict(itertools.islice(D_dicts.items(), 3))
all_patches = get_video_patches(out)

print(len(all_patches))

with open('../preprocessed_pickles/all_patches_check', 'wb') as file:
    pickle.dump(all_patches, file)
