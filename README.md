# ncpsi-hackathon-2024-example
This repository demonstrates how train a YOLO model and standardize predictions for leaderboard.


### Pre-requisites :

- Installing Conda : 

    Install miniconda by following the [official website](https://docs.anaconda.com/miniconda/).

- Check Installation

    ```bash
    conda --version
    >> conda 24.7.1
    ```

- Create a conda environment 

```bash
conda env create -f environment.yml 
```

- Start the environment 
```bash
conda activate example_model
```

- To exit the environment
```bash
conda deactivate
```


>> run the scripts in the conda environment


### Training a yolov8 model


Here are a few steps to prepare your data for training a YOLO model.

- Split your data into ```train``` , ```val``` and ```test```. 

YOLO expects the data to be split into small sub directories 

```bash
    /datasets
    |   /images
    |    |   /train
    |    |   | img1.jpg
    |    |   | img2.jpg
    |    |   /val
    |    |   | imga.jpg
    |    |   | imgb.jpg
    |    |   /test
    |    |   | img00.jpg
    |    |   | img01.jpg
    |    /labels
    |    |   /train
    |    |   | img1.txt
    |    |   | img2.txt
    |    |   /val
    |    |   | imga.txt
    |    |   | imgb.txt
    |    |   /test
    |    |   | img00.txt
    |    |   | img01.txt
    |    plant_data.yaml
```

>> Note : YOLO requires the sub-folders to be named ```datasets```, ```images``` and ```labels```. 


The plant_data.yaml file contains directory paths and class names.
Example :
```yaml
# set paths to train, val and test data.
train: images/train
val: images/val
test: images/test

# list class names
names:
  0: background
  1: Palmer amaranth
  2: Common ragweed
  3: Sicklepod

```

- There are two scripts provided : 

1. ```yolo_train_data.py``` : 

This file trains a YOLOv8 model, on the dataset provided in plant_data.yaml.

Import the model
```python
model = YOLO('yolov8n.pt')
print(model.info())
```


Define some hyperparaters
```python
epochs = 5
batch_size = 5
lr = 0.001
```

Add some optimizers

```python
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
```

Train the model 
```python
model.train(data = plant_data_yml_fpath,
                epochs=epochs,
                batch=batch_size,
                lrf = lr,
                cache=False) 
```

Save the model
```python
model.save('yolov8_example.pt')
```

Read more about training a YOLO model in [1].

2. ```output_yolo_predictions.py```

This file creates a submission file ```<team-name>_predictions.csv```.


Example team1_predictions.csv :

```csv
image_name,class,confidence,x,y,width,height
img1.jpg,7.0,0.7846097350120544,0.6045709252357483,0.8606472015380859,0.10770443081855774,0.27489542961120605
img2.jpg,0.0,0.6409725546836853,0.7783212661743164,0.8416421413421631,0.10649671405553818,0.12297263741493225
img3.jpg,0.0,0.5564624071121216,0.7094271183013916,0.4191618859767914,0.0850377157330513,0.15624332427978516
```


References :
1. https://docs.ultralytics.com/modes/train/#multi-gpu-training
2. Organize your data : https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#option-1-create-a-roboflow-dataset

