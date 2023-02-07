# load environment gnnEnv2
from moviepy.editor import *
import pickle
import numpy as np
from pathlib import Path
import pandas as pd

arg = sys.argv[1]

if arg == "genset":
  out_path = './genset_media_trimmed'    #videos trimmed to motion stop
  in_path = './original_dataset_files/test'                    #original videos and pickle files
  file1 = 'Videos_Test_humanratings'
  file2 = 'genset_video_motionstop_t'
  with open('Human_Experiment_Files/human_rating_labels_genset', 'rb') as hr_file:
    humanratings = pickle.load(hr_file)
else:
  out_path = './media_trimmed'    #videos trimmed to motion stop
  in_path = './original_dataset_files/train'                    #original videos and pickle files
  file1 = 'Videos_humanratings'
  file2 = 'video_motionstop_t'
  with open('Human_Experiment_Files/human_rating_labels', 'rb') as hr_file:
    humanratings = pickle.load(hr_file)

Videos = []
for root, dirs, files, in os.walk(in_path):
  for file in files:
    if file.endswith(".pik"):
      open_f = open(in_path + '/' + file, 'rb')
      open_f_data = pickle.load(open_f)
      name = file[:-3] + 'mp4'

      v = {}
      v['name'] = name
      v['trajectories'] = open_f_data['trajectories']
      v['entity_sizes'] = open_f_data['sizes']
      v['entity_color_code'] = open_f_data['entity_color_code']
      v['physical_goals'] = open_f_data['goals']
      v['wall_segs'] = open_f_data['wall_segs']
      v['landmark_centers'] = open_f_data['landmark_centers']

      v['social_goals'] = humanratings[v['name']]['relationship']

      Videos.append(v)

from collections import Counter
print(Videos[0]['landmark_centers'])
print(Videos[0]['entity_color_code'])
print(Videos[0]['entity_sizes'])
print(Videos[0]['wall_segs'])
print(a)
# Averaging over every 5 trajectory steps
for v in range(len(Videos)):
  traj_v = []
  for e in range(4):
    init = np.array(Videos[v]['trajectories'][e][0])    #initial state
    if len(Videos[v]['trajectories'][e]) % 100 == 0:
        temp = np.array(Videos[v]['trajectories'][e][0:])   #further steps including repeated first step
    else:
        temp = np.array(Videos[v]['trajectories'][e][1:])   #further steps
    traj_avgsteps = np.mean(temp.reshape(-1, 5, 5), axis=1)   #further steps trajectory average over 5
    traj_v.append(np.concatenate((np.expand_dims(init,0),traj_avgsteps)))
  Videos[v]['trajectories'] = traj_v

### Get Motion Stop Time for Each Video
videos_motionstop_all = {}
traj_motionstop_all = {}
for v in range(len(Videos)):
  
  # FIND WHEN MOTION STOPS USING TRAJECTORY INFORMATION
  # Agent 1
  a1 = pd.DataFrame(Videos[v]['trajectories'][0])
  a1_diff = a1.diff() 
  a1_motion_tillstop = a1_diff[(abs(a1_diff)>1e-3).all(axis=1)]  #trajectory till motion stop
  a1_motion_tillstop_t = list(a1_motion_tillstop.index)[-1]   #motion stop time

  # Agent 2
  a2 = pd.DataFrame(Videos[v]['trajectories'][1])
  a2_diff = a2.diff() 
  a2_motion_tillstop = a2_diff[(abs(a2_diff)>1e-3).all(axis=1)]  #trajectory till motion stop
  a2_motion_tillstop_t = list(a2_motion_tillstop.index)[-1] #motion stop time

  motion_stop_t = max(a1_motion_tillstop_t, a2_motion_tillstop_t)
  if motion_stop_t > list(a2.index)[-1]:
    motion_stop_t = list(a2.index)[-1]

  traj_motion_stop_t = motion_stop_t
  clip = VideoFileClip(in_path + '/' + Videos[v]['name'])
  video_motion_stop_t = clip.duration * traj_motion_stop_t / len(Videos[v]['trajectories'][0])
  

  # ADD 2 SECS
  video_motion_stop_t = min(video_motion_stop_t + 2, clip.duration)
  traj_motion_stop_t = int(min(traj_motion_stop_t + len(Videos[v]['trajectories'][0])/clip.duration*2, len(Videos[v]['trajectories'][0])))
  
  videos_motionstop_all[Videos[v]['name']] = video_motion_stop_t
  traj_motionstop_all[Videos[v]['name']] = traj_motion_stop_t


  # Trim trajectories and videos corresponding to new time points
  clip = clip.subclip(0, video_motion_stop_t)
  clip.write_videofile(out_path + '/' + Videos[v]['name'])

  Videos[v]['trajectories'] = [Videos[v]['trajectories'][e][0:traj_motion_stop_t] for e in range(4)]


with open(file1, "wb") as f:
  pickle.dump(Videos, f)

with open(file2, "wb") as f:
  pickle.dump(videos_motionstop_all, f)
  pickle.dump(traj_motionstop_all, f)

