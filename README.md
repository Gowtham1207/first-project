# Weapon Detection System

A deep learning-based weapon detection and classification system that can identify and classify different types of weapons in images.

## Features

- **Weapon Detection**: Detects whether a weapon is present in an image (Yes/No)
- **Weapon Classification**: Classifies the type of weapon (Knife, Handgun, Shotgun, etc.)
- **Confidence Score**: Provides confidence percentage for predictions
- **Clean Web Interface**: Simple and intuitive web interface for predictions

## Dataset

The dataset is organized by weapon type in the `organized_dataset` folder with the following weapon types:
- Automatic Rifle
- Bazooka
- Grenade Launcher
- Handgun
- Knife
- SMG (Submachine Gun)
- Shotgun
- Sniper
- Sword

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Train the model using the training script:

```bash
python train_model.py
```

The training uses:
- **Architecture**: ResNet50 (transfer learning)
- **Epochs**: 20 (adjustable in the script)
- **Batch Size**: 32
- **Learning Rate**: 0.001 with step scheduler

The trained model will be saved in the `models` folder.

### 3. Run the Web Application

After training the model, start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to: `http://127.0.0.1:5000`

## Project Structure

```
weapon_detection/
├── train_model.py                  # Training script
├── predict.py                      # Prediction module
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Web interface HTML
├── organized_dataset/              # Organized dataset
│   ├── train/                     # Training images by weapon type
│   └── val/                       # Validation images by weapon type
├── models/                         # Saved models (created after training)
│   ├── best_weapon_detection_model.pth
│   └── weapon_detection_model_complete.pth
└── uploads/                        # Temporary upload folder (created by Flask app)
```

## Usage

### Training

Run the training script:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the ResNet50 model for 20 epochs
- Save the best model based on validation accuracy
- Generate training history plots
- Generate confusion matrix
- Print classification report

### Prediction (Web Interface)

1. Start the Flask app: `python app.py`
2. Open the web interface in your browser
3. Upload an image (click or drag & drop)
4. Click "Detect Weapon" to get predictions
5. View results:
   - **Weapon Detected**: Yes/No
   - **Weapon Type**: Classification result
   - **Confidence**: Percentage confidence

## Model Architecture

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Transfer Learning**: Early layers frozen, only last layers fine-tuned
- **Output**: 9 weapon classes + dropout for regularization

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP
- Maximum file size: 16MB

## Notes

- The model requires training before use (run `train_model.py`)
- For best results, ensure images are clear and weapons are visible
- Training may take some time depending on your hardware (GPU recommended)
- The web interface will automatically load the trained model on startup
