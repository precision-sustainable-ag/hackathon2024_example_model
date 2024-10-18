import pandas as pd
from ultralytics import YOLO

"""
    This file reads a YOLO model and writes all predictions 
    to a .csv file.
"""

# Load your trained model
model = YOLO('yolov8n.pt')

# Enter file path to test data 
results = model('datasets/images/test')

# Prepare a list to store prediction data
data = []

# Collect results
for result in results:
    image_name = result.path.split('/')[-1]
    for box in result.boxes:
        class_id = box.cls.item()
        conf = box.conf.item()
        x, y, w, h = box.xywh[0].tolist()
        data.append([image_name, class_id, conf, x, y, w, h])

# Convert the list to a DataFrame
df = pd.DataFrame(data, columns=['image_name', 'class', 'confidence', 'x', 'y', 'width', 'height'])

# Output file name
output_csv_fname = "pd_predictions.csv"

# Export results to CSV using pandas
df.to_csv(output_csv_fname, index=False)