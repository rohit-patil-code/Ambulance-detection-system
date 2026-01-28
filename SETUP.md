# 🚑 Ambulance Detection System - Setup Instructions

## Issues Found & Fixes

### Main Issue
The `yolov5` folder is **empty**, which is why the project won't run. This needs to be populated with the YOLOv5 codebase.

## Step-by-Step Setup Guide

### Prerequisites
- Python 3.8 or higher installed
- pip package manager
- Git installed
- Internet connection for downloading dependencies

### Step 1: Navigate to Project Directory
```powershell
cd "C:\Users\rohit\OneDrive\Desktop\code\Ambulance\Ambulance-detection-system"
```

### Step 2: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Clone YOLOv5 Repository
**CRITICAL**: The yolov5 folder is currently empty. You need to clone YOLOv5 into it:

```powershell
# Remove the empty yolov5 folder
Remove-Item -Path "yolov5" -Recurse -Force -ErrorAction SilentlyContinue

# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5.git
```

### Step 4: Install YOLOv5 Dependencies
```powershell
cd yolov5
pip install -r requirements.txt
cd ..
```

### Step 5: Verify Model Weights
Check that `best.pt` exists in the root directory:
```powershell
dir best.pt
```

If `best.pt` is missing, you'll need to either:
- Train your own model on ambulance images, OR
- Obtain a pre-trained model from your source

### Step 6: Run the Application
```powershell
python detection.py
```

### Step 7: Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5001
```

## Common Issues & Solutions

### Issue 1: "YOLOv5 repo not found"
**Solution**: Make sure you cloned YOLOv5 into the `yolov5` folder (Step 3)

### Issue 2: "Model weights not found"
**Solution**: Ensure `best.pt` is in the root directory. If not, you need to train a model or get pre-trained weights.

### Issue 3: "hubconf.py not found"
**Solution**: This means YOLOv5 wasn't properly cloned. Re-run Step 3.

### Issue 4: Import Errors
**Solution**: Install all dependencies:
```powershell
pip install flask torch torchvision opencv-python pillow numpy
```

### Issue 5: Port 5001 Already in Use
**Solution**: Either kill the process using that port, or change the port in `detection.py`:
```python
app.run(debug=True, host='127.0.0.1', port=5002)  # Change to 5002 or another port
```

## Testing the API

### Using curl (if installed):
```powershell
curl -X POST -F "image=@path\to\your\image.jpg" http://127.0.0.1:5001/detect
```

### Using Python:
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post('http://127.0.0.1:5001/detect', files={'image': f})
    print(response.json())
```

## Project Structure After Setup

```
Ambulance-detection-system/
├── detection.py           # Main Flask application
├── draw.py               # Utility functions
├── best.pt               # Model weights (must exist)
├── requirements.txt      # Python dependencies
├── SETUP.md             # This file
├── README.md            # Project documentation
├── templates/
│   └── index.html       # Web interface
├── static/              # Static files
├── yolov5/             # YOLOv5 repository (must be populated)
│   ├── hubconf.py      # Required file
│   ├── models/
│   ├── utils/
│   └── ... (other YOLOv5 files)
└── datasets/           # Training data
```

## Expected Output When Running

When you run `python detection.py`, you should see:
```
Loading YOLOv5 from: C:\Users\rohit\OneDrive\Desktop\code\Ambulance\Ambulance-detection-system\yolov5
Using weights: C:\Users\rohit\OneDrive\Desktop\code\Ambulance\Ambulance-detection-system\best.pt
 * Serving Flask app 'detection'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5001
```

## Quick Start Script (PowerShell)

Save this as `setup.ps1` and run it:

```powershell
# Quick setup script
Write-Host "Setting up Ambulance Detection System..." -ForegroundColor Green

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Clone YOLOv5 if not exists
if (-not (Test-Path "yolov5\hubconf.py")) {
    Write-Host "Cloning YOLOv5..." -ForegroundColor Yellow
    Remove-Item -Path "yolov5" -Recurse -Force -ErrorAction SilentlyContinue
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip install -r requirements.txt
    cd ..
}

# Check for model weights
if (-not (Test-Path "best.pt")) {
    Write-Host "WARNING: best.pt not found! You need model weights." -ForegroundColor Red
} else {
    Write-Host "Model weights found!" -ForegroundColor Green
}

Write-Host "Setup complete! Run 'python detection.py' to start." -ForegroundColor Green
```

## Support

If you encounter any other issues:
1. Check that Python 3.8+ is installed: `python --version`
2. Verify pip is working: `pip --version`
3. Make sure Git is installed: `git --version`
4. Check that all dependencies are installed: `pip list`

## Next Steps After Setup

1. Upload an image containing an ambulance through the web interface
2. The system will detect the ambulance and show a **green** traffic light
3. Upload an image without an ambulance to see a **red** traffic light
4. Use the API endpoint for integration with other systems
