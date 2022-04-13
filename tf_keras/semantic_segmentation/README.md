# Image classification using CNN in Tensorflow/Keras

## steps

#### preprocess images and masks:
```python ./main.py --step=preprocess > ./logs/preprocess.log```

#### patchify images and masks:
```python ./main.py --step=patchify > ./logs/patchify.log```

#### Create tfrecords:
```python ./main.py --step=tfrecord > ./logs/tfrecords.log```

#### load dataset and assess them:
```python ./main.py --step=load_data > ./logs/data_load.log```

#### train the model and assess the model:
```python ./main.py --step=train > ./logs/train.log```

#### fine tune the model and assess the model
```python ./main.py --step=fine_tune > ./logs/fine_tune.log```

#### train and fine tune in one step:
```python ./main.py --step=train_and_fine_tune > ./logs/train_and_fine_tune.log```

#### run predictions:
```python ./main.py --step=predict > ./logs/predict.log```

#### run postprocess:
```python ./main.py --step=postprocess > ./logs/postprocess.log```

 ```--assess``` that is a boolean and can also be added as another parameter. There is an assessment/validation step in code that can be turned off by this parameter.


## Python libraries used in the package
```
python 3.95
tensorflow
matplotlib
pandas
sklearn
albumentations
tqdm
focal-loss
pydot
graphviz
segmentation_models
keras_unet_collection
```