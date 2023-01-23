# Get features batch-wise for PCA
import pickle
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import Model
import sys

with open('../preprocessed_pickles/all_patches', 'rb') as file:
    all_patches = pickle.load(file)

i = int(sys.argv[1])
batch_size = 10000

if batch_size > len(all_patches[i:]):
	batch_size = len(all_patches[i:])

def reshape_patches(x):
	temp = np.expand_dims(x, axis=0)
	temp2 = preprocess_input(smart_resize(temp, (224,224)))
	return temp2[0]

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

import time
start_time = time.time()

all_patches = all_patches[i:i+batch_size]

all_patches_reshaped = [reshape_patches(all_patches[patch_id]) for patch_id in range(batch_size)]
all_patches_reshaped = np.array(all_patches_reshaped)
x = all_patches_reshaped
y = model.predict(x)

file_name = '../preprocessed_pickles/all_features_'+ str(i)
with open(file_name, 'wb') as file:
	pickle.dump(y, file)

print("Time taken: ", time.time()-start_time)