import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

all_features = []
for i in range(0,300000,10000):
	f_name = '../preprocessed_pickles/all_features_'+ str(i)
	with open(f_name, "rb") as f:
		temp = pickle.load(f)
		all_features.extend(temp)


all_features = np.array(all_features)
print(all_features.shape)

# Do PCA
pca = PCA(n_components=20)
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
pca.fit(all_features_scaled)

with open('../preprocessed_pickles/fitted_PCA', "wb") as f:
	pickle.dump(pca,f)
	pickle.dump(scaler,f)

print("PCA on features complete!")
