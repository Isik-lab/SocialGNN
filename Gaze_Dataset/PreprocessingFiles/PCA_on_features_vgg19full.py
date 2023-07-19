# collect all patches, patches = entire frame here
# pass through vgg19 to get features
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import Model
import cv2
import os
import pandas as pd

mode = "pca" #or "features"

if mode == "features":
	def reshape_patches(x):
		temp = np.expand_dims(x, axis=0)
		temp2 = preprocess_input(smart_resize(temp, (224,224)))
		return temp2[0]

	base_model = VGG19(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

	# get annotations to later get only those videos and frames in videos which are annotated
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

	import itertools
	dataset = D_dicts #dict(itertools.islice(D_dicts.items(), 3))

	path = '../video'

	all_features = np.array([])
	for k,v in dataset.items():
	    print("Video:",k)
	    video = cv2.VideoCapture('../video/' + str(k)+ '.mp4')
	    annotations = v

	    print("annotation keys", list(annotations.keys())[-1], len(annotations.keys()))
	    
	    curr_frame = 0

	    all_frames = []
	    while True:
	      ret, frame = video.read()
	  
	      if ret:
	        frame = np.array(frame)
	        if curr_frame not in annotations.keys():
	           curr_frame +=1
	        else:
	            reshaped_frame = np.array(reshape_patches(frame))
	            all_frames.append(reshaped_frame)
	            curr_frame+=1
	      else:
	        break

	    x = np.array(all_frames)
	    y = model.predict(x)
	    print(y.shape)
	    if len(all_features)==0:
               all_features = y
	    else:
	       all_features = np.append(all_features, y, axis=0)

	print(all_features.shape)

	import pickle
	with open('all_features_vgg19full', 'wb') as file:
	    pickle.dump(all_features, file)

elif mode == "pca":
	import pickle
	with open('all_features_vgg19full','rb') as f:
		all_features = pickle.load(f)
	# perform pca on all features (print var expalined by multiple n components)
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler
	# Do PCA
	pca = PCA(n_components=1500)
	scaler = StandardScaler()
	all_features_scaled = scaler.fit_transform(all_features)
	pca.fit(all_features_scaled)
	print(sum(pca.explained_variance_ratio_))

	with open('fitted_PCA_vgg19full', "wb") as f:
		pickle.dump(pca,f)
		pickle.dump(scaler,f)

	print("PCA on features complete!")

	# pick 