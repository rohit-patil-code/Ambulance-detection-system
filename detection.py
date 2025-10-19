from flask import Flask, request, jsonify, render_template
import torch
import cv2
import numpy as np
from PIL import Image
import os
import pathlib

# Windows path compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# Get the directory where detection.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to detection.py
repo_path = os.path.join(BASE_DIR, 'yolov5')
weights_path = os.path.join(BASE_DIR, 'best.pt')

# Validate paths exist
if not os.path.exists(repo_path):
    raise FileNotFoundError(f"YOLOv5 repo not found at: {repo_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at: {weights_path}")
if not os.path.exists(os.path.join(repo_path, 'hubconf.py')):
    raise FileNotFoundError(f"hubconf.py not found in: {repo_path}")

print(f"Loading YOLOv5 from: {repo_path}")
print(f"Using weights: {weights_path}")

# Load the YOLOv5 model
model = torch.hub.load(repo_path, 'custom', path=weights_path, source='local')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def pollTrafficLight():
    try:
        # Read image from request
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = np.array(img)

        # Perform detection
        results = model(img)
        detections = results.xyxy[0].tolist()

        # Check if an ambulance is detected
        ambulance_detected = False
        for *box, conf, cls in detections:
            if conf > 0.80:  # Confidence threshold
                ambulance_detected = True
                break

        return jsonify({"status": "green" if ambulance_detected else "red"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
