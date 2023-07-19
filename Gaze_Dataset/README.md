
The original videos and annotations can be requested from here: [LINK](https://github.com/LifengFan/Human-Gaze-Communication)

Files after preprocessing can be found here: [LINK](https://osf.io/nruvt/?view_only=54d7510958984126acb57b7edb82af8e) (contains the folders: "preprocessed_pickles" and "bootstrapped_traintest_splits_pickles")

To create the train/splits using preprocessed files:
```
python SocialGNN_gazedataset_create_traintest.py standard
```
For VGG19:
```
python SocialGNN_gazedataset_create_traintest.py vgg19fullframe
```
