# 🚑 Ambulance Detection System

A real-time ambulance detection system using YOLOv5 and Flask that helps manage traffic signals for emergency vehicles. The system uses computer vision to detect ambulances and automatically signals for traffic light control.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Custom-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## ✨ Features

- **Real-time Ambulance Detection**: Uses YOLOv5 custom-trained model to detect ambulances in images
- **Traffic Light Integration**: Automatically suggests traffic light status based on ambulance detection
- **Web Interface**: User-friendly web interface with modern, responsive design
- **High Accuracy**: Confidence threshold of 80% for reliable detections
- **REST API**: Easy-to-integrate RESTful API for external applications
- **Cross-platform**: Works on Windows, Linux, and macOS

## 🎯 Demo

The system processes images and returns:
- **Green Signal** 🟢 when an ambulance is detected (confidence > 80%)
- **Red Signal** 🔴 when no ambulance is detected

## 🛠️ Technology Stack

- **Backend Framework**: Flask
- **Deep Learning**: PyTorch, YOLOv5
- **Computer Vision**: OpenCV, PIL
- **Frontend**: HTML5, TailwindCSS, JavaScript
- **Model**: Custom-trained YOLOv5 model

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/rohit-patil-code/Ambulance-detection-system.git
cd Ambulance-detection-system
```

### Step 2: Install Dependencies

```bash
pip install flask torch torchvision opencv-python pillow numpy
```

### Step 3: Download YOLOv5

The YOLOv5 repository should already be included in the `yolov5/` directory. If not, clone it:

```bash
git clone https://github.com/ultralytics/yolov5.git
```

### Step 4: Verify Model Weights

Ensure the `best.pt` file (trained model weights) is present in the root directory. This file contains the custom-trained YOLOv5 model for ambulance detection.

## 🚀 Usage

### Running the Application

1. Start the Flask server:

```bash
python detection.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5001
```

3. Upload an image through the web interface to detect ambulances

### Using the API

Send a POST request to the `/detect` endpoint:

```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://127.0.0.1:5001/detect
```

**Response Format:**
```json
{
  "status": "green"  // or "red"
}
```

## 📁 Project Structure

```
Ambulance-detection-system/
│
├── detection.py           # Main Flask application
├── draw.py               # Utility functions for drawing bounding boxes
├── best.pt               # Custom-trained YOLOv5 model weights
│
├── templates/
│   └── index.html        # Web interface
│
├── static/               # Static files (CSS, JS, images)
│
├── yolov5/              # YOLOv5 repository
│
└── datasets/            # Training/validation datasets
    └── imagenette160/
```

## 🤖 Model Information

- **Architecture**: YOLOv5 (You Only Look Once v5)
- **Training Dataset**: Custom ambulance dataset
- **Confidence Threshold**: 80%
- **Model Weights**: `best.pt` (included in repository)
- **Input**: RGB images
- **Output**: Bounding boxes with class predictions and confidence scores

## 🔌 API Endpoints

### `GET /`
Returns the main web interface.

### `POST /detect`
Detects ambulances in uploaded images.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "status": "green" | "red"
}
```

**Error Response:**
```json
{
  "error": "error message"
}
```

## ⚙️ Configuration

You can modify the following parameters in `detection.py`:

- **Host**: Default is `127.0.0.1`
- **Port**: Default is `5001`
- **Confidence Threshold**: Default is `0.80` (80%)
- **Debug Mode**: Set to `True` for development

```python
# Modify confidence threshold
if conf > 0.80:  # Change this value

# Modify server settings
app.run(debug=True, host='127.0.0.1', port=5001)
```

## 🎨 Web Interface Features

- Modern, responsive design using TailwindCSS
- Drag-and-drop image upload
- Visual traffic light indicator
- Real-time detection results
- Mobile-friendly interface

## 🔍 Troubleshooting

### Common Issues

1. **YOLOv5 not found**
   - Ensure the `yolov5/` directory exists and contains `hubconf.py`

2. **Model weights not found**
   - Verify `best.pt` is in the root directory

3. **Path compatibility issues (Windows)**
   - The code includes Windows path compatibility fixes using `pathlib`

4. **Port already in use**
   - Change the port in `detection.py` or kill the process using port 5001

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 👨‍💻 Author

**Rohit Patil**

- GitHub: [@rohit-patil-code](https://github.com/rohit-patil-code)

## 🙏 Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [Flask](https://flask.palletsprojects.com/) framework
- [PyTorch](https://pytorch.org/) team

---

⭐ If you find this project useful, please consider giving it a star!
