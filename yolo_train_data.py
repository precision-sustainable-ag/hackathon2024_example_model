import torch
from ultralytics import YOLO
import os
"""
    This file trains a yolov8 model
"""
plant_data_yml_fpath = os.path.abspath(os.path.join(os.path.dirname(__file__),'datasets/plant_data.yaml'))

model = YOLO('yolov8n.pt')
print(model.info())

# Define hyperparameters
epochs = 5
batch_size = 5
lr = 0.001

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set model to training mode
model.train(data = plant_data_yml_fpath,
                epochs=epochs,
                batch=batch_size,
                lrf = lr,
                cache=False)  

# Save the trained model
model.save('yolov8_example.pt')

print("Training completed and model saved.")