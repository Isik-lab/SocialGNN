#### Preprocessing for SocialGNN and VisualRNN models
```
conda activate gnnEnv2 (contains opencv)
python get_patches.py
python patches2features.py
python PCA_on_features.py
```
```
python socialgnn_gazedataset_preprocess.py <video_id to start from> 
```
This last command processes only 100 videos at a time, so to process all - run thrice with 0, 100, 200


#### Preprocessing for VGG19 + Linear Layer Model
(This model takes pixel information from the entire frame for each frame, and projected on to 1500 principal components)
```
conda activate gnnEnv2 (contains opencv)
python PCA_on_features_vgg19full.py features
python PCA_on_features_vgg19full.py pca
```

```
python socialgnn_gazedataset_preprocess_vgg19fullframe.py <video_id to start from> 
```
This last command processes only 100 videos at a time, so to process all - run thrice with 0, 100, 200
